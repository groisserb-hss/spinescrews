import math
import csv
import logging
import vtk
import slicer
import qt
import ctk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin


# ---------------------------------------------------------------------------
#  Module-level vector helpers
# ---------------------------------------------------------------------------

def _vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vadd(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _vscale(v, s):
    return [v[0] * s, v[1] * s, v[2] * s]


def _vnorm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vunit(v, eps=1e-9):
    n = _vnorm(v)
    if n < eps:
        return None
    return [v[0] / n, v[1] / n, v[2] / n]


def _fmt3(p):
    return f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}"


# Vertebral levels accepted by the analysis pipeline, caudal-to-cranial.
# MUST stay in sync with `possible_levels` in src/spinescrews/tools/__init__.py
# (this module runs inside 3D Slicer and cannot import the spinescrews package).
POSSIBLE_LEVELS = (
    "LS", "L5", "L4", "L3", "L2", "L1",
    "T13", "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5",
    "T4", "T3", "T2", "T1", "C7", "C6", "C5", "C4", "C3", "C2",
)

# Screw types the pipeline understands (parse_preop_plan), as (label, value)
# pairs.  The value is what gets written to the CSV's screw_type column.
SCREW_TYPES = (
    ("Fixed", "fixed"),
    ("Headless", "headless"),
    ("Polyaxial", "poly"),
    ("Skip (not instrumented)", "skip"),
)
DEFAULT_SCREW_TYPE = "fixed"


def _split_level_side(name):
    """Return ``(level, side)`` if ``name`` is a valid ``<level><side>`` per the
    pipeline (level in POSSIBLE_LEVELS, side L/R, case-sensitive), else ``None``.

    Mirrors the checks in ``parse_preop_plan`` (screw_models.py): a loose
    prefix/suffix test is not enough, since the pipeline KeyErrors on any level
    not in this exact list (e.g. ``S1``, ``L6``, ``T14``)."""
    name = (name or "").strip()
    if len(name) < 2:
        return None
    level, side = name[:-1], name[-1]
    if side in ("L", "R") and level in POSSIBLE_LEVELS:
        return level, side
    return None


# ---------------------------------------------------------------------------
#  1. Module metadata
# ---------------------------------------------------------------------------

class HybridScrewPlanner(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Hybrid Screw Planner"
        self.parent.categories = ["Planning"]
        self.parent.dependencies = []
        self.parent.contributors = ["HSS Consultancy"]
        self.parent.helpText = (
            "Plan orthopedic screws by selecting a Markups Line node. "
            "A cylinder model is auto-generated along the line trajectory "
            "at the specified length and radius. Entry point (first control "
            "point) is preserved; the tip is re-snapped to the chosen length."
        )
        self.parent.acknowledgementText = ""


# ---------------------------------------------------------------------------
#  2. Widget (UI + events)
# ---------------------------------------------------------------------------

class HybridScrewPlannerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = HybridScrewPlannerLogic()

        self._observedLineNode = None
        self._updating = False

        self.updateTimer = qt.QTimer()
        self.updateTimer.setSingleShot(True)
        self.updateTimer.setInterval(25)
        self.updateTimer.timeout.connect(self.onAutoUpdateTimeout)

        # ---- Instructions (collapsed) ----
        instructionsButton = ctk.ctkCollapsibleButton()
        instructionsButton.text = "Instructions"
        instructionsButton.collapsed = True
        self.layout.addWidget(instructionsButton)
        instructionsLayout = qt.QVBoxLayout(instructionsButton)

        intro = qt.QLabel(
            "Workflow:\n"
            "1) Create/place a Markups Line yourself (entry first, tip second)\n"
            "2) Select it below\n"
            "3) Set desired screw length + radius\n"
            "4) The cylinder preview will auto-update as you move the line\n\n"
            "Behavior:\n"
            "- First point stays where you place it\n"
            "- Current trajectory is preserved\n"
            "- Second point is re-snapped to the chosen length\n"
            "- Cylinder model updates from the line\n"
            "- Markup line remains the interaction object"
        )
        intro.wordWrap = True
        instructionsLayout.addWidget(intro)

        # ---- Parameters (open) ----
        parametersButton = ctk.ctkCollapsibleButton()
        parametersButton.text = "Parameters"
        self.layout.addWidget(parametersButton)
        parametersLayout = qt.QFormLayout(parametersButton)

        self.lineSelector = slicer.qMRMLNodeComboBox()
        self.lineSelector.nodeTypes = ["vtkMRMLMarkupsLineNode"]
        self.lineSelector.noneEnabled = True
        self.lineSelector.addEnabled = False
        self.lineSelector.removeEnabled = False
        self.lineSelector.renameEnabled = False
        self.lineSelector.editEnabled = False
        self.lineSelector.setMRMLScene(slicer.mrmlScene)
        parametersLayout.addRow("Input line:", self.lineSelector)

        self.lengthSpin = qt.QDoubleSpinBox()
        self.lengthSpin.minimum = 1.0
        self.lengthSpin.maximum = 300.0
        self.lengthSpin.decimals = 2
        self.lengthSpin.value = 40.0
        self.lengthSpin.suffix = " mm"
        parametersLayout.addRow("Desired length:", self.lengthSpin)

        self.radiusSpin = qt.QDoubleSpinBox()
        self.radiusSpin.minimum = 0.1
        self.radiusSpin.maximum = 20.0
        self.radiusSpin.decimals = 2
        self.radiusSpin.value = 2.5
        self.radiusSpin.suffix = " mm"
        parametersLayout.addRow("Cylinder radius:", self.radiusSpin)

        self.screwTypeCombo = qt.QComboBox()
        for label, value in SCREW_TYPES:
            self.screwTypeCombo.addItem(label, value)
        parametersLayout.addRow("Screw type:", self.screwTypeCombo)

        self.opacitySpin = qt.QDoubleSpinBox()
        self.opacitySpin.minimum = 0.05
        self.opacitySpin.maximum = 1.0
        self.opacitySpin.decimals = 2
        self.opacitySpin.singleStep = 0.05
        self.opacitySpin.value = 1.0
        parametersLayout.addRow("Model opacity:", self.opacitySpin)

        self.keepLineVisibleCheck = qt.QCheckBox()
        self.keepLineVisibleCheck.checked = True
        parametersLayout.addRow("Keep markup line visible:", self.keepLineVisibleCheck)

        self.modelColorButton = qt.QPushButton("Set cylinder color")
        self.currentColor = qt.QColor(255, 255, 255)
        self.modelColorButton.setStyleSheet(
            f"background-color: {self.currentColor.name()}"
        )
        self.modelColorButton.clicked.connect(self.onPickColor)
        parametersLayout.addRow("Cylinder display:", self.modelColorButton)

        # ---- Actions (open) ----
        actionsButton = ctk.ctkCollapsibleButton()
        actionsButton.text = "Actions"
        self.layout.addWidget(actionsButton)
        actionsLayout = qt.QVBoxLayout(actionsButton)

        buttonRow = qt.QHBoxLayout()
        actionsLayout.addLayout(buttonRow)

        self.applyButton = qt.QPushButton("Apply/Update now")
        self.deleteModelButton = qt.QPushButton("Delete cylinder for selected line")
        self.exportButton = qt.QPushButton("Export all line coordinates...")
        buttonRow.addWidget(self.applyButton)
        buttonRow.addWidget(self.deleteModelButton)
        buttonRow.addWidget(self.exportButton)

        self.statusLabel = qt.QLabel("Ready.")
        actionsLayout.addWidget(self.statusLabel)

        self.logBox = qt.QPlainTextEdit()
        self.logBox.readOnly = True
        self.logBox.setMinimumHeight(170)
        actionsLayout.addWidget(self.logBox)

        # Stretch at bottom so widgets stay at top of panel
        self.layout.addStretch(1)

        # ---- Connections ----
        self.applyButton.clicked.connect(self.onApplyManual)
        self.deleteModelButton.clicked.connect(self.onDeleteModel)
        self.exportButton.clicked.connect(self.onExportAllLines)

        try:
            self.lineSelector.connect(
                "currentNodeChanged(vtkMRMLNode*)",
                self.onSelectedLineChanged,
            )
        except Exception as e:
            logging.warning(f"Could not connect lineSelector signal: {e}")

        self.lengthSpin.valueChanged.connect(self.onParametersChanged)
        self.radiusSpin.valueChanged.connect(self.onParametersChanged)
        self.screwTypeCombo.currentIndexChanged.connect(self.onParametersChanged)
        self.opacitySpin.valueChanged.connect(self.onParametersChanged)
        self.keepLineVisibleCheck.toggled.connect(self.onParametersChanged)

        # ---- Scene-close observers (via VTKObservationMixin) ----
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.StartCloseEvent,
            self.onSceneStartClose,
        )
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.EndCloseEvent,
            self.onSceneEndClose,
        )

        # Kick off initial state
        self.onSelectedLineChanged(self.lineSelector.currentNode())

    # -- Module lifecycle --------------------------------------------------

    def enter(self):
        """Called when the user navigates to this module."""
        if self._observedLineNode is not None:
            # Check the node is still in the scene
            if slicer.mrmlScene.GetNodeByID(self._observedLineNode.GetID()):
                self._startObservingLine(self._observedLineNode)
                self.scheduleAutoUpdate()
            else:
                self._observedLineNode = None

    def exit(self):
        """Called when the user switches away from this module."""
        self._stopObservingLine(clearNodeReference=False)
        self.updateTimer.stop()

    def cleanup(self):
        """Full teardown when the module widget is destroyed."""
        self.updateTimer.stop()
        # Disconnect Qt signals
        self.applyButton.clicked.disconnect(self.onApplyManual)
        self.deleteModelButton.clicked.disconnect(self.onDeleteModel)
        self.exportButton.clicked.disconnect(self.onExportAllLines)
        self.lengthSpin.valueChanged.disconnect(self.onParametersChanged)
        self.radiusSpin.valueChanged.disconnect(self.onParametersChanged)
        self.screwTypeCombo.currentIndexChanged.disconnect(self.onParametersChanged)
        self.opacitySpin.valueChanged.disconnect(self.onParametersChanged)
        self.keepLineVisibleCheck.toggled.disconnect(self.onParametersChanged)
        self.modelColorButton.clicked.disconnect(self.onPickColor)
        try:
            self.lineSelector.disconnect(
                "currentNodeChanged(vtkMRMLNode*)",
                self.onSelectedLineChanged,
            )
        except Exception:
            pass
        # Remove all VTK observers (scene + markup) via mixin
        self.removeObservers()

    # -- Scene-close handlers ----------------------------------------------

    def onSceneStartClose(self, caller, event):
        self._stopObservingLine(clearNodeReference=True)
        self.updateTimer.stop()

    def onSceneEndClose(self, caller, event):
        self._observedLineNode = None
        self.statusLabel.setText("Ready.")

    # -- Observer management (via VTKObservationMixin) ----------------------

    def _onPointModified(self, caller, event, callData=None):
        if self._updating:
            return
        self.scheduleAutoUpdate()

    def _onPointEndInteraction(self, caller, event, callData=None):
        if self._updating:
            return
        self.updateFromSelectedLine(logUpdate=False)

    def _startObservingLine(self, lineNode):
        if lineNode is None:
            return
        if not self.hasObserver(
            lineNode,
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
            self._onPointModified,
        ):
            self.addObserver(
                lineNode,
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                self._onPointModified,
            )
        if not self.hasObserver(
            lineNode,
            slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
            self._onPointEndInteraction,
        ):
            self.addObserver(
                lineNode,
                slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                self._onPointEndInteraction,
            )

    def _stopObservingLine(self, clearNodeReference=True):
        if self._observedLineNode is not None:
            self.removeObserver(
                self._observedLineNode,
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                self._onPointModified,
            )
            self.removeObserver(
                self._observedLineNode,
                slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent,
                self._onPointEndInteraction,
            )
        if clearNodeReference:
            self._observedLineNode = None

    # -- Slots / callbacks -------------------------------------------------

    def _appendLog(self, text):
        self.logBox.appendPlainText(text)
        logging.info(text)

    def onPickColor(self):
        picked = qt.QColorDialog.getColor(self.currentColor, self.parent)
        if picked.isValid():
            self.currentColor = picked
            self.modelColorButton.setStyleSheet(
                f"background-color: {self.currentColor.name()}"
            )
            self.scheduleAutoUpdate()

    def onParametersChanged(self, *args):
        self.scheduleAutoUpdate()

    def _currentScrewType(self):
        """Canonical screw_type value (e.g. 'fixed') for the current dropdown selection."""
        value = self.screwTypeCombo.itemData(self.screwTypeCombo.currentIndex)
        return value if value else DEFAULT_SCREW_TYPE

    def onSelectedLineChanged(self, node=None):
        self._stopObservingLine(clearNodeReference=True)
        self._observedLineNode = node

        if node is None:
            self.statusLabel.setText("No line selected.")
            return

        self._loadSettingsFromLine(node)
        self._startObservingLine(node)
        self.statusLabel.setText(f"Watching line: {node.GetName()}")
        self.scheduleAutoUpdate()

    def _loadSettingsFromLine(self, lineNode):
        """Load a line's stored screw settings back into the controls, so
        revisiting a line shows its own values instead of silently overwriting
        them on the next auto-update.  No-op if the line has no model yet."""
        modelNode = self.logic.getAssociatedModelNode(lineNode)
        if modelNode is None:
            return  # never applied -> keep current control values for this line

        # Suppress onParametersChanged -> scheduleAutoUpdate while we populate.
        self._updating = True
        try:
            for attr, spin in (
                ("ScrewLengthMm", self.lengthSpin),
                ("ScrewRadiusMm", self.radiusSpin),
            ):
                stored = modelNode.GetAttribute(attr)
                if stored:
                    spin.value = float(stored)
            storedType = modelNode.GetAttribute("ScrewType")
            if storedType:
                idx = self.screwTypeCombo.findData(storedType)
                if idx >= 0:
                    self.screwTypeCombo.setCurrentIndex(idx)
        finally:
            self._updating = False

    def scheduleAutoUpdate(self):
        if self._updating:
            return
        self.updateTimer.start()

    def onAutoUpdateTimeout(self):
        self.updateFromSelectedLine(logUpdate=False)

    def onApplyManual(self):
        self.updateFromSelectedLine(logUpdate=True)

    # -- Core update (thin wrapper around Logic) ---------------------------

    def updateFromSelectedLine(self, logUpdate=False):
        lineNode = self.lineSelector.currentNode()
        if not lineNode:
            self.statusLabel.setText("Select a line first.")
            return

        color = (
            self.currentColor.redF(),
            self.currentColor.greenF(),
            self.currentColor.blueF(),
        )

        self._updating = True
        try:
            with slicer.util.tryWithErrorDisplay("Failed to update screw model."):
                result = self.logic.updateScrewFromLine(
                    lineNode,
                    length=float(self.lengthSpin.value),
                    radius=float(self.radiusSpin.value),
                    screw_type=self._currentScrewType(),
                    color=color,
                    opacity=float(self.opacitySpin.value),
                    keepLineVisible=bool(self.keepLineVisibleCheck.checked),
                )
        finally:
            self._updating = False

        if result["status"] == "no_points":
            self.statusLabel.setText("The selected line needs 2 defined control points.")
            return

        if result["status"] == "zero_length":
            self.statusLabel.setText(
                "Line length is zero. Move the tip away from the entry."
            )
            return

        self.statusLabel.setText("Auto-updated from selected line.")

        if logUpdate:
            self._appendLog("=== Applied/Updated ===")
            self._appendLog(f"Line: {result['lineName']}")
            self._appendLog(f"Model: {result['modelName']}")
            self._appendLog(f"Entry RAS: {result['entry']}")
            self._appendLog(f"Adjusted tip RAS: {result['tip']}")
            self._appendLog(f"Desired length (mm): {result['desiredLength']:.3f}")
            self._appendLog(f"Final line length (mm): {result['finalLength']:.3f}")
            self._appendLog(f"Cylinder radius (mm): {result['radius']:.3f}")
            self._appendLog(f"Screw type: {result['screwType']}")

    # -- Export / Delete (delegate to Logic, handle UI here) ---------------

    def onExportAllLines(self):
        rows, skipped, warnings = self.logic.gatherExportRows()

        if not rows:
            self.statusLabel.setText("No valid line nodes found to export.")
            self._appendLog("Export aborted: no line nodes with 2 defined points found.")
            return

        filePath = qt.QFileDialog.getSaveFileName(
            self.parent,
            "Export line coordinates",
            "screw_line_coordinates.csv",
            "CSV Files (*.csv)",
        )
        if isinstance(filePath, tuple):
            filePath = filePath[0]
        if not filePath:
            self.statusLabel.setText("Export cancelled.")
            return

        with slicer.util.tryWithErrorDisplay("Failed to export CSV."):
            with open(filePath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "line_name",
                    "screw_type",
                    "line_id",
                    "entry_ras_x",
                    "entry_ras_y",
                    "entry_ras_z",
                    "tip_ras_x",
                    "tip_ras_y",
                    "tip_ras_z",
                    "length_mm",
                    "cylinder_radius_mm",
                    "cylinder_model_name",
                ])
                writer.writerows(rows)

            self._appendLog(f"Exported {len(rows)} line(s) to: {filePath}")

            if skipped:
                self._appendLog(
                    "Skipped line(s) without 2 defined points: "
                    + ", ".join(skipped)
                )

            if warnings:
                self.statusLabel.setText(
                    f"Exported {len(rows)} line(s) -- {len(warnings)} "
                    f"consistency warning(s), see log."
                )
                self._appendLog(
                    f"--- {len(warnings)} pipeline-consistency warning(s) ---"
                )
                for w in warnings:
                    self._appendLog("WARNING: " + w)
            else:
                self.statusLabel.setText(f"Exported {len(rows)} line(s).")

    def onDeleteModel(self):
        lineNode = self.lineSelector.currentNode()
        if not lineNode:
            self.statusLabel.setText("Select a line first.")
            return

        with slicer.util.tryWithErrorDisplay("Failed to delete cylinder model."):
            deleted = self.logic.deleteModelForLine(lineNode)

        if not deleted:
            self.statusLabel.setText("No associated cylinder model found.")
            return

        self.statusLabel.setText("Associated cylinder deleted.")
        self._appendLog(f"Deleted cylinder model for line: {lineNode.GetName()}")


# ---------------------------------------------------------------------------
#  3. Logic (pure computation + MRML, no UI state)
# ---------------------------------------------------------------------------

class HybridScrewPlannerLogic(ScriptedLoadableModuleLogic):
    MODEL_ID_ATTR = "HybridScrewModelNodeID"

    def getLineEndpointsWorld(self, lineNode):
        if not lineNode:
            return None
        if lineNode.GetNumberOfDefinedControlPoints() < 2:
            return None

        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]

        ok0 = lineNode.GetLineStartPositionWorld(p0)
        ok1 = lineNode.GetLineEndPositionWorld(p1)
        if not ok0 or not ok1:
            return None

        return p0, p1

    def makeCylinderPolyData(self, start, end, radius, sides=24):
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(start)
        lineSource.SetPoint2(end)

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(lineSource.GetOutputPort())
        tube.SetRadius(radius)
        tube.SetNumberOfSides(sides)
        tube.CappingOn()
        tube.Update()

        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(tube.GetOutput())
        return polyData

    def getAssociatedModelNode(self, lineNode):
        modelNodeID = lineNode.GetAttribute(self.MODEL_ID_ATTR)
        if not modelNodeID:
            return None
        return slicer.mrmlScene.GetNodeByID(modelNodeID)

    def createModelNode(self, lineNode, color, opacity):
        baseName = lineNode.GetName() if lineNode.GetName() else "ScrewLine"
        modelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode",
            f"{baseName}_Cylinder",
        )
        modelNode.CreateDefaultDisplayNodes()

        displayNode = modelNode.GetDisplayNode()
        displayNode.SetColor(color[0], color[1], color[2])
        displayNode.SetOpacity(opacity)
        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)
        displayNode.SetVisibility3D(True)

        lineNode.SetAttribute(self.MODEL_ID_ATTR, modelNode.GetID())
        modelNode.SetAttribute("SourceMarkupLineID", lineNode.GetID())
        return modelNode

    def updateModelDisplay(self, modelNode, color, opacity):
        displayNode = modelNode.GetDisplayNode()
        if not displayNode:
            return
        displayNode.SetColor(color[0], color[1], color[2])
        displayNode.SetOpacity(opacity)
        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)
        displayNode.SetVisibility3D(True)

    def updateScrewFromLine(self, lineNode, length, radius, color, opacity,
                            keepLineVisible, screw_type=DEFAULT_SCREW_TYPE):
        """Snap the tip, create/update the cylinder. Always returns a dict
        with a ``'status'`` key: ``'no_points'``, ``'zero_length'``, or ``'ok'``."""
        endpoints = self.getLineEndpointsWorld(lineNode)
        if endpoints is None:
            return {"status": "no_points"}

        entryPoint, currentTip = endpoints
        direction = _vsub(currentTip, entryPoint)
        unitDirection = _vunit(direction)

        if unitDirection is None:
            return {"status": "zero_length"}

        adjustedTip = _vadd(entryPoint, _vscale(unitDirection, length))

        lineNode.SetLineStartPositionWorld(entryPoint)
        lineNode.SetLineEndPositionWorld(adjustedTip)

        finalLength = lineNode.GetLineLengthWorld()

        polyData = self.makeCylinderPolyData(entryPoint, adjustedTip, radius)

        modelNode = self.getAssociatedModelNode(lineNode)
        if modelNode is None:
            modelNode = self.createModelNode(lineNode, color, opacity)

        modelNode.SetAndObservePolyData(polyData)
        self.updateModelDisplay(modelNode, color, opacity)

        modelNode.SetAttribute("ScrewEntryRAS", _fmt3(entryPoint))
        modelNode.SetAttribute("ScrewTipRAS", _fmt3(adjustedTip))
        modelNode.SetAttribute("ScrewLengthMm", f"{length:.6f}")
        modelNode.SetAttribute("ScrewRadiusMm", f"{radius:.6f}")
        modelNode.SetAttribute("ScrewType", screw_type)

        if lineNode.GetDisplayNode():
            lineNode.GetDisplayNode().SetVisibility(keepLineVisible)

        return {
            "status": "ok",
            "lineName": lineNode.GetName(),
            "modelName": modelNode.GetName(),
            "entry": list(entryPoint),
            "tip": list(adjustedTip),
            "desiredLength": length,
            "finalLength": finalLength,
            "radius": radius,
            "screwType": screw_type,
        }

    def gatherExportRows(self):
        """Return ``(rows, skipped, warnings)`` for every Markups Line in the scene.

        ``rows`` are pipeline-ready CSV records (column order matches the header
        written in ``onExportAllLines``); ``skipped`` lists lines without 2
        defined points; ``warnings`` flag content that ``parse_preop_plan`` would
        reject -- bad/missing level names, unpaired L/R sides, and lines exported
        without a cylinder model (empty radius / defaulted screw type)."""
        lineNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")

        rows = []
        skipped = []
        warnings = []
        sides_by_level = {}

        for lineNode in sorted(lineNodes, key=lambda n: n.GetName().lower()):
            endpoints = self.getLineEndpointsWorld(lineNode)
            if endpoints is None:
                skipped.append(lineNode.GetName())
                continue

            entryPoint, tipPoint = endpoints
            lengthMm = lineNode.GetLineLengthWorld()
            name = lineNode.GetName()

            modelNode = self.getAssociatedModelNode(lineNode)
            modelName = ""
            radiusMm = ""
            screwType = DEFAULT_SCREW_TYPE

            if modelNode is None:
                warnings.append(
                    f"'{name}': no cylinder model -- exported with empty radius "
                    f"and default type '{DEFAULT_SCREW_TYPE}'. Apply it first."
                )
            else:
                modelName = modelNode.GetName()
                radiusAttr = modelNode.GetAttribute("ScrewRadiusMm")
                if radiusAttr is not None:
                    radiusMm = radiusAttr
                typeAttr = modelNode.GetAttribute("ScrewType")
                if typeAttr:
                    screwType = typeAttr
                else:
                    warnings.append(
                        f"'{name}': no screw type stored; exported as "
                        f"'{DEFAULT_SCREW_TYPE}'."
                    )

            # Name / level validity -- the pipeline KeyErrors on bad levels.
            parsed = _split_level_side(name)
            if parsed is None:
                warnings.append(
                    f"'{name}': not a valid <level><side> name (level must be "
                    f"one of {POSSIBLE_LEVELS[0]}..{POSSIBLE_LEVELS[-1]}, side "
                    f"L or R); the pipeline will reject it."
                )
            else:
                level, side = parsed
                sides_by_level.setdefault(level, []).append(side)

            rows.append([
                name,
                screwType,
                lineNode.GetID(),
                f"{entryPoint[0]:.6f}",
                f"{entryPoint[1]:.6f}",
                f"{entryPoint[2]:.6f}",
                f"{tipPoint[0]:.6f}",
                f"{tipPoint[1]:.6f}",
                f"{tipPoint[2]:.6f}",
                f"{lengthMm:.6f}",
                radiusMm,
                modelName,
            ])

        # L/R pairing -- the pipeline requires both sides per level
        # (screw_models.py:104-108); a skip-type line is the way to fill an
        # un-instrumented side.
        for level in sorted(sides_by_level, key=POSSIBLE_LEVELS.index):
            sides = sides_by_level[level]
            for side in ("L", "R"):
                count = sides.count(side)
                if count == 0:
                    warnings.append(
                        f"level {level}: missing {side} side; the pipeline "
                        f"requires paired L+R (add a {level}{side} line, type "
                        f"'skip' if that side is not instrumented)."
                    )
                elif count > 1:
                    warnings.append(
                        f"level {level}: {side} side appears {count} times; "
                        f"expected exactly one."
                    )

        return rows, skipped, warnings

    def deleteModelForLine(self, lineNode):
        """Remove the associated cylinder model. Return True if deleted."""
        modelNode = self.getAssociatedModelNode(lineNode)
        if modelNode is None:
            return False

        slicer.mrmlScene.RemoveNode(modelNode)
        lineNode.SetAttribute(self.MODEL_ID_ATTR, "")
        return True


# ---------------------------------------------------------------------------
#  4. Tests
# ---------------------------------------------------------------------------

class HybridScrewPlannerTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_UpdateScrewFromLine()
        self.setUp()
        self.test_GatherExportRows()
        self.setUp()
        self.test_ScrewTypeRoundTrip()
        self.setUp()
        self.test_NameValidation()
        self.setUp()
        self.test_DeleteModelForLine()

    def test_UpdateScrewFromLine(self):
        logic = HybridScrewPlannerLogic()

        # Test with a properly placed line
        lineNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", "TestLine"
        )
        lineNode.AddControlPoint([0, 0, 0])
        lineNode.AddControlPoint([0, 0, 50])

        result = logic.updateScrewFromLine(
            lineNode,
            length=40,
            radius=2.5,
            color=(1, 1, 1),
            opacity=1.0,
            keepLineVisible=True,
        )
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["desiredLength"], 40.0)

        # Verify cylinder model was created
        modelNode = logic.getAssociatedModelNode(lineNode)
        self.assertIsNotNone(modelNode)
        self.assertIsNotNone(modelNode.GetPolyData())

        # Test with a line that has no points (should return no_points)
        emptyLine = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", "EmptyLine"
        )
        result = logic.updateScrewFromLine(
            emptyLine,
            length=40,
            radius=2.5,
            color=(1, 1, 1),
            opacity=1.0,
            keepLineVisible=True,
        )
        self.assertEqual(result["status"], "no_points")

        self.delayDisplay("test_UpdateScrewFromLine passed")

    def _addModeledLine(self, logic, name, screw_type="fixed", start=(0, 0, 0),
                        end=(0, 0, 50)):
        """Create a 2-point Markups line and apply a screw model + type to it."""
        line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
        line.AddControlPoint(list(start))
        line.AddControlPoint(list(end))
        logic.updateScrewFromLine(
            line, length=40, radius=2.5, color=(1, 1, 1), opacity=1.0,
            keepLineVisible=True, screw_type=screw_type,
        )
        return line

    def test_GatherExportRows(self):
        logic = HybridScrewPlannerLogic()

        # Two valid, paired, modeled lines (T11 L+R)
        self._addModeledLine(logic, "T11L", "fixed", start=(10, 20, 30), end=(40, 50, 60))
        self._addModeledLine(logic, "T11R", "poly", start=(0, 0, 0), end=(0, 0, 100))

        # A line with insufficient points -> skipped
        slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "IncompleteLine")

        rows, skipped, warnings = logic.gatherExportRows()
        self.assertEqual(len(rows), 2)
        self.assertIn("IncompleteLine", skipped)

        # 12 columns; screw_type is column index 1
        self.assertEqual(len(rows[0]), 12)
        types_by_name = {r[0]: r[1] for r in rows}
        self.assertEqual(types_by_name["T11L"], "fixed")
        self.assertEqual(types_by_name["T11R"], "poly")

        # Valid + paired + modeled -> nothing for the pipeline to complain about
        self.assertEqual(warnings, [])

        self.delayDisplay("test_GatherExportRows passed")

    def test_ScrewTypeRoundTrip(self):
        logic = HybridScrewPlannerLogic()

        # All four canonical types, kept paired so pairing stays clean.
        cases = [("T10L", "fixed"), ("T10R", "headless"),
                 ("T9L", "poly"), ("T9R", "skip")]
        for i, (name, stype) in enumerate(cases):
            result = self._addModeledLine(
                logic, name, stype, start=(0, 0, float(i)), end=(0, 0, float(i) + 50)
            )
            # updateScrewFromLine echoes the type back
            modelNode = logic.getAssociatedModelNode(result)
            self.assertEqual(modelNode.GetAttribute("ScrewType"), stype)

        rows, skipped, warnings = logic.gatherExportRows()
        types_by_name = {r[0]: r[1] for r in rows}
        for name, stype in cases:
            self.assertEqual(types_by_name[name], stype)
        self.assertEqual(warnings, [])

        self.delayDisplay("test_ScrewTypeRoundTrip passed")

    def test_NameValidation(self):
        logic = HybridScrewPlannerLogic()

        self._addModeledLine(logic, "T11L")       # valid, paired below
        self._addModeledLine(logic, "T11R")
        self._addModeledLine(logic, "S1L")        # invalid level
        self._addModeledLine(logic, "L1L")        # valid level, missing R partner

        rows, skipped, warnings = logic.gatherExportRows()
        joined = " | ".join(warnings)

        self.assertIn("S1L", joined)              # bad level flagged
        self.assertIn("level L1", joined)         # unpaired side flagged
        self.assertNotIn("level T11", joined)     # fully paired -> not flagged

        self.delayDisplay("test_NameValidation passed")

    def test_DeleteModelForLine(self):
        logic = HybridScrewPlannerLogic()

        # Create a line and a screw model
        lineNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", "DeleteTestLine"
        )
        lineNode.AddControlPoint([0, 0, 0])
        lineNode.AddControlPoint([0, 0, 50])

        result = logic.updateScrewFromLine(
            lineNode,
            length=40,
            radius=2.5,
            color=(1, 1, 1),
            opacity=1.0,
            keepLineVisible=True,
        )
        self.assertEqual(result["status"], "ok")

        # Verify model exists
        modelNode = logic.getAssociatedModelNode(lineNode)
        self.assertIsNotNone(modelNode)

        # Delete and verify removal
        deleted = logic.deleteModelForLine(lineNode)
        self.assertTrue(deleted)

        modelNode = logic.getAssociatedModelNode(lineNode)
        self.assertIsNone(modelNode)

        # Deleting again should return False
        deleted = logic.deleteModelForLine(lineNode)
        self.assertFalse(deleted)

        self.delayDisplay("test_DeleteModelForLine passed")
