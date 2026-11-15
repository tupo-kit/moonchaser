"""Integrated Qt shell for the VTK globe: search, points, analysis controls, top bar and comparison tables."""
import csv
import os
import queue
import re
import sys
import threading
import time
from datetime import timedelta

import numpy as np

# QVTK's X11 render window needs a real X11 window id. Under native Wayland Qt
# hands out a pointer-sized id, and VTK fails with "The result is out of range"
# followed by an X BadWindow error. So on a Wayland session Qt must run through
# XWayland (the "xcb" platform). This deliberately OVERRIDES an inherited
# QT_QPA_PLATFORM=wayland, which many desktop sessions export. To opt out, set
# MOON_KEEP_QT_PLATFORM=1.
if sys.platform.startswith('linux') and not os.environ.get('MOON_KEEP_QT_PLATFORM'):
    _wayland = (bool(os.environ.get('WAYLAND_DISPLAY'))
                or os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland')
    if _wayland and os.environ.get('DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    elif _wayland:
        print('Moon Horizon: Wayland session without XWayland (DISPLAY is not set). '
              'The VTK window needs XWayland; install/enable it.', file=sys.stderr)

import vtk

from PySide6 import QtCore, QtGui, QtWidgets
QT_BINDING = 'PySide6'

import vtkmodules.qt
vtkmodules.qt.PyQtImpl = QT_BINDING
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import compare
import hud
import sky
import terrain
import visibility_search as search

Qt = QtCore.Qt

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
EDIT_FORMAT = 'yyyy-MM-dd HH:mm:ss'
MONO_FAMILY = 'monospace'
RESULTS_HEIGHT = 300
RESULTS_BAR = 49
QWIDGETSIZE_MAX = 16777215
TABLES_HEIGHT = 260
ABOVE_ONLY, ALL_SAMPLES = 'Above horizon only', 'All samples'
UNIT_SECONDS = {'min': 60.0, 'h': 3600.0, 'd': 86400.0}
CELL_NUMBER = re.compile(r'([+-]?[\d,]*\.?\d+)\s*(min|h|d)?\*?')

STYLE = '''
QWidget { background: transparent; color: #e6edf5; font-size: 13px; }
QMainWindow, QDialog { background: #131b28; }
QFrame#header { background: #0f1622; border-bottom: 1px solid #26334a; }
QFrame#sidebar { background: #1a2433; border-right: 1px solid #26334a; }
QFrame#tabRow { border-bottom: 1px solid #26334a; }
QFrame#popup { background: #1a2433; border: 1px solid #34445a; border-radius: 6px; }
#mono, QDateTimeEdit, QTreeWidget, QTableWidget { font-family: "MONO"; }
QLabel#brand { font-size: 14px; font-weight: 600; }
QLabel#groupTitle { font-weight: 600; }
QLabel#muted { color: #9fb0c3; font-size: 12px; }
QLabel#error { color: #ff8f85; }
QPushButton, QToolButton { background: #222e40; border: 1px solid #34445a; border-radius: 6px; padding: 0 12px; min-height: 30px; color: #e6edf5; }
QPushButton:hover, QToolButton:hover { background: #2b3a50; }
QPushButton:pressed, QPushButton:checked, QToolButton:checked { background: #2e3c50; }
QToolButton:checked { background: #446078; border-color: #79a6c1; }
QFrame#timeline { background: #111a29; border: 1px solid #293951; border-radius: 7px; }
QFrame#timeline QSlider::groove:horizontal { height: 6px; border: 1px solid #42526a; background: #273448; border-radius: 3px; }
QFrame#timeline QSlider::handle:horizontal { background: #27bbd0; border: 2px solid #e6f8ff; width: 12px; margin: -5px 0; border-radius: 3px; }
QPushButton:disabled { color: #5f6f83; background: #1c2635; border-color: #26334a; }
QPushButton::menu-indicator { image: none; width: 0; }
QPushButton#primary { background: #2bb5c5; border-color: #2bb5c5; color: #06222a; font-weight: 600; }
QPushButton#primary:hover { background: #4cc6d4; }
QPushButton#primary:disabled { background: #1d4a52; border-color: #1d4a52; color: #5f8f96; }
QPushButton#ghost { background: transparent; border-color: transparent; color: #9fb0c3; }
QPushButton#ghost:hover { background: #222e40; }
QLineEdit, QComboBox, QDateTimeEdit { background: #0f1622; color: #e6edf5; border: 1px solid #34445a; border-radius: 6px; padding: 0 10px; min-height: 30px; selection-background-color: #1b3a47; }
QLineEdit:focus, QComboBox:focus, QDateTimeEdit:focus { border-color: #2bb5c5; }
QLineEdit:disabled { color: #5f6f83; }
QLineEdit[error="true"] { border-color: #ff6b5e; }
QComboBox::drop-down { border: 0; width: 22px; }
QComboBox QAbstractItemView { background: #222e40; border: 1px solid #34445a; selection-background-color: #2e3c50; outline: 0; }
QTreeWidget { background: #1a2433; border: 0; alternate-background-color: #1e2a3b; outline: 0; }
QTreeWidget::item:selected { background: #1b3a47; color: #e6edf5; }
QHeaderView::section { background: #1a2433; color: #9fb0c3; border: 0; border-bottom: 1px solid #26334a; padding: 6px 10px; font-weight: 500; }
QTabWidget::pane { border: 0; }
QTabWidget::tab-bar { alignment: left; }
QTabBar::tab { background: transparent; color: #9fb0c3; padding: 0 14px; min-height: 32px; border: 0; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #e6edf5; border-bottom-color: #2bb5c5; }
QTabBar#sideTabs::tab { min-height: 40px; font-weight: 500; }
QScrollArea { border: 0; }
QFrame#pointRow { border-radius: 8px; }
QFrame#pointRow[on="true"] { background: #222e40; }
QFrame#card { background: #1a2433; border: 1px solid #2e3c50; border-radius: 8px; }
QFrame#barTop { border-top: 1px solid #26334a; }
QFrame#results { background: #1a2433; border-top: 1px solid #26334a; }
QFrame#topBar { border-bottom: 1px solid #26334a; }
QFrame#analysisHeader { background: #111a29; border: 1px solid #293951; border-top-left-radius: 7px; border-top-right-radius: 7px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; }
QLineEdit#pointSearch:focus { border-color: #d8ac4e; }
QFrame#tables { background: #131b28; border-top: 1px solid #26334a; }
QFrame#staleBox[stale="true"] { background: #3a2f14; border: 1px solid #7a6220; border-radius: 8px; }
QFrame#staleBox[stale="empty"] { background: #222c39; border: 1px solid #465568; border-radius: 8px; }
QFrame#staleBox[stale="empty"] QLabel#warning { color: #b0bfd0; }
QLabel#warning { color: #f5e3a8; }
QTableWidget { background: #1a2433; alternate-background-color: #1e2a3b; border: 0; outline: 0; }
QTableWidget::item { padding: 0 10px; }
QTableWidget::item:selected { background: #1b3a47; }
QFrame#seg { background: #0f1622; border-radius: 7px; }
QPushButton#seg { background: transparent; border: 0; border-radius: 5px; padding: 0; min-height: 26px; color: #9fb0c3; }
QPushButton#seg:hover { background: #222e40; }
QPushButton#seg:disabled { background: transparent; color: #5f6f83; }
QPushButton#seg:checked { background: #2e3c50; color: #e6edf5; font-weight: 500; }
QToolButton#icon { background: transparent; border: 0; padding: 0; min-height: 30px; min-width: 30px; }
QToolButton#icon:hover { background: #2b3a50; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #34445a; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:horizontal { background: #34445a; border-radius: 4px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QProgressBar { border: 0; border-radius: 3px; background: #0f1622; max-height: 6px; }
QProgressBar::chunk { background: #2bb5c5; border-radius: 3px; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #5d708a; border-radius: 4px; background: #0f1622; }
QCheckBox::indicator:checked { background: #2bb5c5; border-color: #2bb5c5; }
QCheckBox::indicator:hover { border-color: #e6edf5; }
QCheckBox:disabled { color: #5f6f83; }
QCheckBox::indicator:disabled { background: #1c2635; border-color: #34445a; }
QCheckBox::indicator:checked:disabled { background: #3b6b73; border-color: #3b6b73; }
QRadioButton { spacing: 8px; }
QRadioButton::indicator { width: 16px; height: 16px; border: 1px solid #5d708a; border-radius: 9px; background: #0f1622; }
QRadioButton::indicator:checked { border-color: #2bb5c5; background: qradialgradient(cx: .5, cy: .5, radius: .5, fx: .5, fy: .5, stop: 0 #2bb5c5, stop: .45 #2bb5c5, stop: .55 #0f1622, stop: 1 #0f1622); }
QRadioButton::indicator:disabled { border-color: #34445a; background: #1c2635; }
QSplitter::handle { background: #26334a; }
QMenu { background: #1a2433; color: #e6edf5; border: 1px solid #34445a; border-radius: 6px; padding: 4px; }
QMenu::item { padding: 6px 24px 6px 24px; border-radius: 4px; }
QMenu::item:selected { background: #2e3c50; }
QMenu::item:disabled { color: #5f6f83; }
QMenu::separator { height: 1px; background: #26334a; margin: 4px 6px; }
QToolTip { background: #222e40; color: #e6edf5; border: 1px solid #34445a; padding: 4px; }
'''
class GlobeWidget(QVTKRenderWindowInteractor):
    # On macOS every buffer swap re-exposes the widget, so rendering on each
    # paint event repaints forever and starves the event loop.
    _paint_pending = True

    def resizeEvent(self, event):
        self._paint_pending = True
        super().resizeEvent(event)

    def paintEvent(self, event):
        if self._paint_pending:
            self._paint_pending = False
            super().paintEvent(event)


class Entry(QtWidgets.QLineEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setObjectName('mono')
    def get(self): return self.text()
    def delete(self, *_): self.clear()
    def insert(self, _index, value): self.setText(str(value))
    def configure(self, **kw):
        if 'state' in kw: self.setEnabled(kw['state'] != 'disabled')


class Label(QtWidgets.QLabel):
    def configure(self, **kw):
        if 'text' in kw: self.setText(str(kw['text']))
        if 'state' in kw: self.setEnabled(kw['state'] != 'disabled')


class StatusLabel(Label):
    def __init__(self, *args):
        super().__init__(*args)
        self.setObjectName('muted')
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
    def setText(self, text):
        self.setToolTip(text)
        super().setText(text)
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.WindowText))
        painter.drawText(self.rect(), Qt.AlignLeft | Qt.AlignVCenter,
                         self.fontMetrics().elidedText(self.text(), Qt.ElideRight, self.width()))


class Segmented(QtWidgets.QFrame):
    changed = QtCore.Signal(str)
    def __init__(self, values):
        super().__init__()
        self.setObjectName('seg')
        self.value = values[0]
        self.buttons = {}
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(2)
        group = QtWidgets.QButtonGroup(self)
        for value in values:
            button = QtWidgets.QPushButton(value)
            button.setObjectName('seg')
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, v=value: self._clicked(v))
            group.addButton(button)
            row.addWidget(button, 1)
            self.buttons[value] = button
        self.buttons[self.value].setChecked(True)
    def _clicked(self, value):
        if value != self.value:
            self.value = value
            self.changed.emit(value)
    def get(self): return self.value
    def set(self, value):
        self.value = value
        self.buttons[value].setChecked(True)
    def configure(self, **kw):
        if 'values' in kw:
            for value, button in self.buttons.items(): button.setVisible(value in kw['values'])


class TerrainSwitch(Segmented):
    def __init__(self):
        super().__init__(('Quick search', 'Precise search'))
        super().set('Precise search')
    def get(self): return self.value == 'Precise search'
    def set(self, value): super().set('Precise search' if value else 'Quick search')


class Gate:
    def __init__(self, target, needs=None):
        self.target, self.needs, self.allowed = target, needs, False
        target.setEnabled(False)
    def configure(self, **kw):
        if 'state' in kw:
            self.allowed = kw['state'] != 'disabled'
            self.apply()
    def apply(self):
        self.target.setEnabled(self.allowed and (self.needs is None or self.needs()))


class Button(QtWidgets.QPushButton):
    def configure(self, **kw):
        if 'text' in kw: self.setText(str(kw['text']))
        if 'state' in kw: self.setEnabled(kw['state'] != 'disabled')


class CancelButton(Button):
    def __init__(self, text, *companions):
        super().__init__(text)
        self.companions = companions
        self.configure(state='disabled')
    def configure(self, **kw):
        if 'state' in kw:
            running = kw['state'] != 'disabled'
            self.setVisible(running)
            for widget in self.companions: widget.setVisible(running)


class Check(QtWidgets.QCheckBox):
    def get(self): return self.isChecked()
    def set(self, value): self.setChecked(bool(value))
    def configure(self, **kw):
        if 'state' in kw: self.setEnabled(kw['state'] != 'disabled')

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked():
            return
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        rect = self.style().subElementRect(QtWidgets.QStyle.SE_CheckBoxIndicator, option, self)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        color = QtGui.QColor('#06222a' if self.isEnabled() else '#a6b8c0')
        pen = QtGui.QPen(color)
        pen.setWidthF(2.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        painter.drawPolyline(QtGui.QPolygonF([
            QtCore.QPointF(x + w * .22, y + h * .52),
            QtCore.QPointF(x + w * .43, y + h * .72),
            QtCore.QPointF(x + w * .79, y + h * .29),
        ]))
        painter.end()


class Combo(QtWidgets.QComboBox):
    def get(self): return self.currentText()
    def set(self, value):
        self.blockSignals(True)
        self.setCurrentText(str(value))
        self.blockSignals(False)
    def configure(self, **kw):
        if 'values' in kw:
            old = self.currentText()
            self.blockSignals(True)
            self.clear()
            self.addItems(list(kw['values']))
            if old in kw['values']: self.setCurrentText(old)
            self.blockSignals(False)


class Progress(QtWidgets.QProgressBar):
    def configure(self, **kw):
        if 'maximum' in kw: self.setMaximum(int(kw['maximum']))
        if 'value' in kw: self.setValue(int(kw['value']))


class Tree(QtWidgets.QTreeWidget):
    """Small adapter allowing the existing search result logic to populate Qt."""
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.keys = list(columns)
        self.callbacks = {}
        self.tag_colors = {}
        self.setColumnCount(len(columns))
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.header().setSectionsClickable(True)
        self.header().setStretchLastSection(False)
        for index in range(1, len(self.keys)):
            self.headerItem().setTextAlignment(index, Qt.AlignRight | Qt.AlignVCenter)
        self.header().sectionClicked.connect(self._heading_clicked)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
    def _heading_clicked(self, index):
        callback = self.callbacks.get(self.keys[index])
        if callback: callback()
    def configure(self, **kw):
        if 'displaycolumns' in kw:
            shown = set(kw['displaycolumns'])
            for index, key in enumerate(self.keys): self.setColumnHidden(index, key not in shown)
    def heading(self, key, **kw):
        index = self.keys.index(key)
        if 'text' in kw: self.headerItem().setText(index, str(kw['text']))
        if 'command' in kw: self.callbacks[key] = kw['command']
    def column(self, key, **kw):
        if 'width' in kw: self.setColumnWidth(self.keys.index(key), int(kw['width']))
    def get_children(self):
        return tuple(str(self.topLevelItem(i).data(0, Qt.UserRole)) for i in range(self.topLevelItemCount()))
    def delete(self, *ids):
        if ids: self.clear_rows()
    def clear_rows(self):
        previous = self.blockSignals(True)
        try:
            self.clear()
        finally:
            self.blockSignals(previous)
    def select_first(self, count):
        if count <= 0:
            self.clearSelection()
            return
        model = self.selectionModel()
        first, last = self.model().index(0, 0), self.model().index(count - 1, self.columnCount() - 1)
        selection = QtCore.QItemSelection(first, last)
        model.select(selection, QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Rows)
    def tag_configure(self, tag, **kw):
        if 'foreground' in kw:
            color = QtGui.QColor(kw['foreground'])
            self.tag_colors[tag] = color.lighter(175)
    def insert(self, _parent, _where, iid, tags=(), values=()):
        item = QtWidgets.QTreeWidgetItem([str(value) for value in values])
        item.setData(0, Qt.UserRole, str(iid))
        for tag in tags:
            if tag in self.tag_colors:
                brush = QtGui.QBrush(self.tag_colors[tag])
                for i in range(len(self.keys)): item.setForeground(i, brush)
        for i in range(1, len(self.keys)):
            if self.keys[i] != 'cell': item.setTextAlignment(i, Qt.AlignRight | Qt.AlignVCenter)
        self.addTopLevelItem(item)
    def selection_count(self):
        return sum(item.bottom() - item.top() + 1 for item in self.selectionModel().selection())
    def selection(self):
        # Item ids are display row indices, so avoid scanning the entire table.
        return tuple(sorted((str(item.data(0, Qt.UserRole)) for item in self.selectedItems()), key=int))
    def selection_set(self, ids):
        if isinstance(ids, str): ids = (ids,)
        previous = self.blockSignals(True)
        try:
            self.clearSelection()
            for iid in ids:
                index = int(iid)
                if 0 <= index < self.topLevelItemCount():
                    item = self.topLevelItem(index)
                    if str(item.data(0, Qt.UserRole)) == str(iid):
                        item.setSelected(True)
        finally:
            self.blockSignals(previous)
        if not previous:
            self.itemSelectionChanged.emit()
    def see(self, iid):
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if str(item.data(0, Qt.UserRole)) == str(iid): self.scrollToItem(item); break


class TabAdapter:
    def __init__(self, select): self.select = select


class WindowAdapter:
    def __init__(self, panel): self.panel = panel
    def after(self, ms, callback):
        timer = QtCore.QTimer(self.panel.widget)
        timer.setSingleShot(True)
        def fired():
            try: callback()
            finally: timer.deleteLater()
        timer.timeout.connect(fired)
        timer.start(ms)
        return timer
    def after_cancel(self, timer):
        if timer is not None: timer.stop(); timer.deleteLater()


def group(title, layout):
    box = QtWidgets.QWidget()
    stack = QtWidgets.QVBoxLayout(box)
    stack.setContentsMargins(0, 0, 0, 0)
    stack.setSpacing(8)
    label = QtWidgets.QLabel(title)
    label.setObjectName('groupTitle')
    stack.addWidget(label)
    stack.addLayout(layout)
    return box


def muted_label(text):
    label = QtWidgets.QLabel(text)
    label.setObjectName('muted')
    return label


def counted(count, noun):
    return f'{count:,} {noun}{"" if count == 1 else "s"}'


def cell_value(text):
    match = CELL_NUMBER.fullmatch(text.strip())
    return float(match[1].replace(',', '')) * UNIT_SECONDS.get(match[2], 1.0) if match else None


def dot_icon(color):
    pixmap = QtGui.QPixmap(20, 20)
    pixmap.fill(Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setBrush(color)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QtCore.QPointF(10, 10), 6, 6)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QtGui.QIcon(pixmap)


def form_layout():
    form = QtWidgets.QFormLayout()
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return form


def labeled(layout, title, widget):
    label = QtWidgets.QLabel(title)
    label.setObjectName('muted')
    layout.addRow(label, widget)


class SearchPanel(search.SearchDialog):
    """Existing search engine and result behavior, hosted in a docked Qt panel."""
    def __init__(self, shell, owner):
        self.shell, self.owner = shell, owner
        self.widget = QtWidgets.QWidget()
        self.window = WindowAdapter(self)
        self.fetch, self.cache = owner.fetch_skyfield, owner._point_cache
        self.on_close, self.on_apply = None, owner.apply_visibility_search
        self.existing_points = lambda: list(owner._points)
        self._closed = False
        self.messages, self.cancel = queue.Queue(), threading.Event()
        self._fill_cancel = threading.Event()
        self._fill_generation = 0
        self._fill_running = self._fill_dirty = False
        self._fill_refreshed = 0.0
        self._tick = 0
        self.worker = self.timer = None
        self.results, self.selected, self.selected_points = [], None, []
        self.add_to_compare = self.show_matches = self.clear_markers = self.analyze = False
        self.marker_results, self.marker_mode = None, 'Points'
        self._display_items, self.last_parameters = [], None
        self._sort = self._cluster_cache = None
        self.finished = self.was_cancelled = False
        self._pairing = False
        self._area_preview_enabled = True
        self._area_search_running = False
        self._area_search_bounds = None
        self._graph_seen = False
        self._reopen_results = False
        self.results_collapsed = False
        self.results_height = RESULTS_HEIGHT
        self.workspace_tabs = TabAdapter(self._select_workspace)
        self.workspace_settings, self.workspace_results = 'settings', 'results'
        self.settings_tabs, self.area_tab = TabAdapter(lambda _page: None), None
        outer = QtWidgets.QVBoxLayout(self.widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._build_criteria(outer)
        self._build_results()
        shell.right.addWidget(self.results_panel)
        shell.right.setStretchFactor(1, 0)
        self._update_lunar_state()
        self._refresh_headings()
        if owner._search_session is not None: self.restore(owner._search_session)
        self.poll()

    def _build_criteria(self, outer):
        scroll = self.criteria_scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(body)
        stack.setContentsMargins(16, 16, 16, 16)
        stack.setSpacing(18)
        self.f = {}
        for name, value in zip(('start', 'end'), self.shell.interval):
            entry = Entry(value.strftime(TIME_FORMAT), self.widget)
            entry.hide()
            self.f[name] = entry
        self.use_terrain = TerrainSwitch()
        mode = QtWidgets.QVBoxLayout()
        mode.setSpacing(8)
        mode.addWidget(self.use_terrain)
        note = QtWidgets.QLabel('Precise search uses the terrain horizon and returns points. Quick search ignores terrain and returns candidate clusters.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        mode.addWidget(note)
        stack.addLayout(mode)
        visibility = QtWidgets.QGridLayout()
        visibility.setSpacing(8)
        self._pair_row(visibility, 0, 'Earth %', Entry('50'), Entry('100'), 'min_earth', 'max_earth')
        self._pair_row(visibility, 1, 'Sun %', Entry('30'), Entry('100'), 'min_sun', 'max_sun')
        stack.addWidget(group('Visibility', visibility))
        area = QtWidgets.QGridLayout()
        area.setSpacing(8)
        for key, value in (('lat', '-90, 90'), ('lon', '-180, 180')):
            hidden = Entry(value, self.widget)
            hidden.hide()
            self.f[key] = hidden
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = Entry('-90'), Entry('90'), Entry('-180'), Entry('180')
        self._pair_row(area, 0, 'Latitude', self.lat_min, self.lat_max)
        self._pair_row(area, 1, 'Longitude', self.lon_min, self.lon_max)
        self._link_pair('lat', self.lat_min, self.lat_max)
        self._link_pair('lon', self.lon_min, self.lon_max)
        for row, (key, label, value) in enumerate((('spacing', 'Grid, deg', '10'), ('step', 'Time step', '2')), 2):
            entry = Entry(value)
            self.f[key] = entry
            area.addWidget(self._muted(label), row, 0)
            area.addWidget(entry, row, 1)
        self.step_unit = Combo()
        self.step_unit.addItems(['seconds', 'minutes', 'hours', 'days'])
        self.step_unit.setCurrentText('hours')
        self.f['step_unit'] = self.step_unit
        area.addWidget(self.step_unit, 3, 3)
        self.area_box = group('Area', area)
        area_layout = self.area_box.layout()
        area_title = area_layout.takeAt(0).widget()
        area_heading = QtWidgets.QHBoxLayout(); area_heading.addWidget(area_title); area_heading.addStretch(1)
        self.draw_area_button = QtWidgets.QToolButton()
        self.draw_area_button.setIcon(line_icon('area')); self.draw_area_button.setIconSize(QtCore.QSize(18, 18))
        self.draw_area_button.setCheckable(True)
        self.draw_area_button.setToolTip('Draw a search area on the map; right-click to cancel')
        self.draw_area_button.setAccessibleName('Draw search area on map')
        self.draw_area_button.clicked.connect(self.shell.toggle_area_tool)
        self.show_area = QtWidgets.QCheckBox('Show area')
        self.show_area.setChecked(True)
        self.show_area.toggled.connect(lambda _: self.shell.sync_search_area())
        area_heading.addWidget(self.show_area)
        area_heading.addWidget(self.draw_area_button); area_layout.insertLayout(0, area_heading)
        stack.addWidget(self.area_box)
        self._area_timer = QtCore.QTimer(self.widget); self._area_timer.setSingleShot(True)
        self._area_timer.timeout.connect(self.shell.sync_search_area)
        for field in (self.lat_min, self.lat_max, self.lon_min, self.lon_max):
            field.textChanged.connect(self._area_edited)
        stack.addWidget(self._build_advanced())
        stack.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        for name in ('start', 'end'): self.f[name].textChanged.connect(self._interval_edited)
        self.lunar_avg_check.clicked.connect(self._update_lunar_state)
        self.lunar_sd_check.clicked.connect(self._update_lunar_state)
        self.use_terrain.changed.connect(self._update_lunar_state)
        bottom = QtWidgets.QFrame()
        bottom.setObjectName('barTop')
        pinned = QtWidgets.QVBoxLayout(bottom)
        pinned.setContentsMargins(16, 12, 16, 16)
        pinned.setSpacing(8)
        self.status = StatusLabel('Choose criteria and start the search.')
        pinned.addWidget(self.status)
        self.progress = Progress()
        self.progress.setTextVisible(False)
        pinned.addWidget(self.progress)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = Button('Start search')
        self.start_button.setObjectName('primary')
        self.start_button.setFixedHeight(36)
        self.start_button.clicked.connect(self.start)
        self.cancel_button = CancelButton('Cancel', self.progress)
        self.cancel_button.setFixedHeight(36)
        self.cancel_button.clicked.connect(self.cancel.set)
        buttons.addWidget(self.start_button, 1)
        buttons.addWidget(self.cancel_button)
        pinned.addLayout(buttons)
        outer.addWidget(bottom)

    def _build_advanced(self):
        box = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(box)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(18)
        toggle = QtWidgets.QPushButton()
        toggle.setObjectName('ghost')
        toggle.setCheckable(True)
        toggle.setFixedHeight(28)
        row = QtWidgets.QHBoxLayout(toggle)
        row.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel('Advanced')
        title.setObjectName('groupTitle')
        summary = QtWidgets.QLabel('Slope · Screening · Lunar-day statistics ›')
        summary.setObjectName('muted')
        for label in (title, summary): label.setAttribute(Qt.WA_TransparentForMouseEvents)
        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(summary)
        stack.addWidget(toggle)
        content = QtWidgets.QWidget()
        inner = QtWidgets.QVBoxLayout(content)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(18)
        def field(form, key, label, value):
            entry = Entry(str(value))
            self.f[key] = entry
            labeled(form, label, entry)
        f = form_layout()
        field(f, 'max_slope', 'Maximum slope (deg; optional)', '')
        field(f, 'slope_radius', 'Slope radius (km)', '1')
        inner.addWidget(group('Slope', f))
        f = form_layout()
        field(f, 'side', 'Samples per side', '3')
        field(f, 'horizon', 'Pessimistic horizon (deg)', '2')
        self.screen_mode = Combo()
        self.screen_mode.addItems(['Pessimistic center', 'All cells'])
        self.range_mode = Combo()
        self.range_mode.addItems(['On', 'Off'])
        labeled(f, 'Cell screening', self.screen_mode)
        labeled(f, 'Optimistic range', self.range_mode)
        field(f, 'workers', 'Worker processes (0 = auto, 1 = off)', '0')
        note = QtWidgets.QLabel('Screening is approximate and can miss suitable cells.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        f.addRow(note)
        inner.addWidget(group('Screening', f))
        f = form_layout()
        self.lunar_avg_check = self.lunar_avg = Check('Calculate and show avg Sun visible % per lunar day')
        self.lunar_sd_check = self.lunar_sd = Check('Calculate and show standard deviation')
        self.lunar_sd_check.setToolTip('Enable the average first. Available for terrain searches longer than two days.')
        f.addRow(self.lunar_avg_check)
        f.addRow(self.lunar_sd_check)
        for key, label in [('lunar_min_avg', 'Minimum average %'), ('lunar_max_avg', 'Maximum average %'),
                           ('lunar_min_sd', 'Minimum standard deviation %'), ('lunar_max_sd', 'Maximum standard deviation %')]: field(f, key, label, '')
        self.lunar_note = Label()
        self.lunar_note.setObjectName('muted')
        self.lunar_note.setWordWrap(True)
        f.addRow(self.lunar_note)
        inner.addWidget(group('Lunar-day statistics', f))
        self.details = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(self.details)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self.fill_label, self.cache_label, self.range_label, self.horizon_label = Label(), Label(), Label(), Label()
        for label in (self.fill_label, self.cache_label, self.range_label, self.horizon_label):
            label.setObjectName('muted')
            label.setWordWrap(True)
            detail_layout.addWidget(label)
        self.details.hide()
        details_button = QtWidgets.QPushButton('Calculation details')
        details_button.setCheckable(True)
        details_button.toggled.connect(self.details.setVisible)
        inner.addWidget(details_button)
        inner.addWidget(self.details)
        content.hide()
        stack.addWidget(content)
        toggle.toggled.connect(content.setVisible)
        toggle.toggled.connect(lambda on: summary.setText('Slope · Screening · Lunar-day statistics ' + ('⌄' if on else '›')))
        return box

    def _muted(self, text):
        label = QtWidgets.QLabel(text)
        label.setObjectName('muted')
        return label

    def _pair_row(self, grid, row, title, low, high, low_key=None, high_key=None):
        if low_key:
            self.f[low_key], self.f[high_key] = low, high
        grid.setColumnMinimumWidth(0, 90)
        grid.setColumnMinimumWidth(2, 12)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.addWidget(self._muted(title), row, 0)
        grid.addWidget(low, row, 1)
        grid.addWidget(self._muted('–'), row, 2)
        grid.addWidget(high, row, 3)

    def _link_pair(self, key, low, high):
        hidden = self.f[key]
        def compose():
            if self._pairing: return
            self._pairing = True
            hidden.setText(f'{low.text()}, {high.text()}')
            self._pairing = False
        def split(text):
            parts = text.replace(';', ',').split(',')
            if self._pairing or len(parts) != 2: return
            self._pairing = True
            low.setText(parts[0].strip())
            high.setText(parts[1].strip())
            self._pairing = False
        low.textChanged.connect(compose)
        high.textChanged.connect(compose)
        hidden.textChanged.connect(split)

    def _interval_edited(self):
        self.shell.set_interval(self.f['start'].get(), self.f['end'].get())
        self._update_lunar_state()

    def _build_results(self):
        panel = self.results_panel = QtWidgets.QFrame()
        panel.setObjectName('results')
        panel.hide()
        root = QtWidgets.QVBoxLayout(panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        header = QtWidgets.QFrame()
        header.setObjectName('tabRow')
        header.setFixedHeight(48)
        bar = QtWidgets.QHBoxLayout(header)
        bar.setContentsMargins(14, 0, 14, 0)
        bar.setSpacing(10)
        title = QtWidgets.QLabel('Results')
        title.setObjectName('groupTitle')
        bar.addWidget(title)
        self.count_label = QtWidgets.QLabel()
        self.count_label.setObjectName('muted')
        bar.addWidget(self.count_label)
        self.view_mode = Segmented(('Points', 'Clusters'))
        self.view_mode.setFixedSize(170, 32)
        self.view_mode.changed.connect(lambda _: self.populate_results('Showing') if self.results else None)
        bar.addSpacing(8)
        bar.addWidget(self.view_mode)
        bar.addStretch(1)
        self.columns_button = QtWidgets.QPushButton('Columns')
        self.columns_button.setObjectName('ghost')
        self.columns_button.clicked.connect(self._show_columns)
        bar.addWidget(self.columns_button)
        selected = lambda: self.table.selectionModel().hasSelection()
        one = lambda: self.table.selection_count() == 1
        compare = Button('Analyze')
        compare.clicked.connect(self.analyze_selected)
        add = Button('Add to analysis list')
        add.setObjectName('primary')
        add.clicked.connect(self.add_selected)
        more = QtWidgets.QPushButton('⋯')
        more.setAccessibleName('More actions')
        self.more_menu = QtWidgets.QMenu(self.shell)
        more.setMenu(self.more_menu)
        self.collapse_button = QtWidgets.QPushButton('⌄')
        self.collapse_button.setObjectName('ghost')
        self.collapse_button.setAccessibleName('Collapse results')
        self.collapse_button.clicked.connect(lambda: self.set_results_collapsed(not self.results_collapsed))
        for widget in (compare, add, more, self.collapse_button): bar.addWidget(widget)
        self.analyze_button, self.add_button = Gate(compare, selected), Gate(add, selected)
        self.view_button = self._menu_action('Show selected on globe', self.show_selected, selected)
        self.select_all_button = self._menu_action('Select all', self.select_all)
        self.marks_button = self._menu_action('Show all on globe', self.show_all)
        self.more_menu.addSeparator()
        self.export_button = self._menu_action('Save CSV', self.export)
        self.more_menu.addAction('Clear markers', self.clear)
        self.gates = [self.analyze_button, self.add_button, self.view_button, self.select_all_button,
                      self.marks_button, self.export_button]
        root.addWidget(header)
        self._columns = tuple(spec[0] for spec in search.COLUMN_SPECS)
        self._heading_labels = {spec[0]: spec[1] for spec in search.COLUMN_SPECS}
        self.column_vars = {}
        self._build_columns_popup()
        self.table = Tree(self._columns)
        for spec in search.COLUMN_SPECS: self.table.column(spec[0], width=spec[3])
        self.table.itemDoubleClicked.connect(lambda *_: self.view_selected())
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(
            lambda pos: self.more_menu.popup(self.table.viewport().mapToGlobal(pos)))
        self.results_body = self.table
        root.addWidget(self.table, 1)

    def _menu_action(self, text, callback, needs=None):
        action = self.more_menu.addAction(text, callback)
        return Gate(action, needs)

    def _build_columns_popup(self):
        popup = self.columns_popup = QtWidgets.QFrame(self.shell, Qt.Popup)
        popup.setObjectName('popup')
        layout = QtWidgets.QGridLayout(popup)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setHorizontalSpacing(24)
        index = 0
        for spec in search.COLUMN_SPECS:
            checkbox = Check(spec[1])
            checkbox.set(spec[0] in search.DEFAULT_COLUMNS)
            checkbox.clicked.connect(self._columns_changed)
            self.column_vars[spec[0]] = checkbox
            if spec[0] in search.PERCENT_COLUMNS:
                layout.addWidget(checkbox, index // 2, index % 2)
                index += 1

    def _show_columns(self):
        popup = self.columns_popup
        popup.adjustSize()
        button = self.columns_button
        point = button.mapToGlobal(QtCore.QPoint(button.width() - popup.width(), -popup.height() - 4))
        popup.move(point)
        popup.show()

    def _selection_changed(self):
        for gate in self.gates: gate.apply()
        self._update_count()

    def _update_count(self):
        shown = bool(self.results)
        if shown and self.results_panel.isHidden() and not self.shell._zen and not self.owner._graph_expanded:
            self.results_panel.show()
            if not self.results_collapsed: QtCore.QTimer.singleShot(0, self._open_height)
        elif not shown:
            self.results_panel.hide()
            self.count_label.setText('')
            return
        clustered = self.view_mode.get() == 'Clusters'
        total = len(self._clusters()) if clustered else len(self.results)
        noun = 'cluster' if clustered else 'point'
        selected = self.table.selection_count()
        self.count_label.setText(counted(total, noun) + (f' · {selected} selected' if selected else ''))

    def populate_results(self, label):
        super().populate_results(label)
        self._update_count()

    def _area_edited(self, *_):
        if not self._area_search_running:
            self._area_preview_enabled = True
        self._area_timer.start(160)

    def area_bounds(self):
        if self._area_search_running and self._area_search_bounds is not None:
            return self._area_search_bounds
        try:
            lat0, lat1, lon0, lon1 = (float(field.text()) for field in
                                       (self.lat_min, self.lat_max, self.lon_min, self.lon_max))
        except ValueError:
            return None
        if not (-90 <= lat0 < lat1 <= 90 and lon0 < lon1):
            return None
        return lat0, lat1, lon0, min(lon1, lon0 + 360.0)

    def start(self):
        if self.owner._area_drawing:
            self.shell.cancel_area_tool()
        if self._area_search_running and self.worker is not None and not self.worker.is_alive():
            self._area_search_running = False
        bounds = self.area_bounds()
        previous_worker = self.worker
        super().start()
        if self.worker is not None and self.worker is not previous_worker:
            self.owner.show_visibility_markers([])
            self.owner._search_selected = None
            self.owner.window.Render()
            self._area_search_bounds = bounds
            self._area_search_running = True
            self._area_preview_enabled = True
            self.shell.sync_search_area()
        self._update_count()

    def poll(self):
        super().poll()
        if self._area_search_running and self.worker is not None and not self.worker.is_alive():
            self._area_search_running = False
            self._area_preview_enabled = False
            self.shell.sync_search_area()

    def _select_workspace(self, page):
        if page == self.workspace_results:
            if self.results and not self.owner._graph_expanded:
                self.set_results_collapsed(False)
        else:
            self.shell.show_panel('search')

    def set_results_collapsed(self, collapsed):
        if collapsed and not self.results_collapsed:
            self.results_height = max(self.results_panel.height(), RESULTS_BAR)
        self.results_collapsed = collapsed
        self.results_body.setVisible(not collapsed)
        self.collapse_button.setText('⌃' if collapsed else '⌄')
        self.results_panel.setMaximumHeight(RESULTS_BAR if collapsed else QWIDGETSIZE_MAX)
        if self.results_panel.isVisible():
            QtCore.QTimer.singleShot(0, self._open_height)

    def set_zen(self, hidden):
        if hidden:
            self.results_panel.hide()
        else:
            self._update_count()

    def _open_height(self):
        split = self.shell.right
        height = RESULTS_BAR if self.results_collapsed else self.results_height
        sizes = split.sizes()
        total = sum(sizes)
        sizes[1] = height
        sizes[0] = max(1, total - height - sizes[2])
        split.setSizes(sizes)

    def sync_graph(self, expanded):
        if expanded == self._graph_seen:
            return
        self._graph_seen = expanded
        if expanded:
            self.results_panel.hide()
        else:
            self._update_count()

    def export(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self.widget,'Save results','','CSV files (*.csv)')
        if path:
            if not path.lower().endswith('.csv'): path += '.csv'
            self.export_to(path)

    def close(self):
        if self._closed: return
        self._closed=True; self.cancel.set(); self._fill_cancel.set()
        self.window.after_cancel(self.timer)
        self.widget.hide()


class TimePanel(QtWidgets.QWidget):
    def __init__(self, shell, owner):
        super().__init__()
        self.shell, self.owner = shell, owner
        self.graph_rows = []
        self.graph_key = None
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(18)
        sampling = QtWidgets.QVBoxLayout()
        sampling.setSpacing(8)
        title = QtWidgets.QLabel('Sampling')
        title.setObjectName('groupTitle')
        sampling.addWidget(title)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 90)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self.step = Entry()
        self.step.setObjectName('mono')
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(['seconds', 'minutes', 'hours', 'days'])
        self.method = Segmented(tuple(sky.SKY_METHODS))
        grid.addWidget(muted_label('Step'), 0, 0)
        grid.addWidget(self.step, 0, 1)
        grid.addWidget(self.unit, 0, 2)
        grid.addWidget(muted_label('Model'), 1, 0)
        grid.addWidget(self.method, 1, 1, 1, 2)
        sampling.addLayout(grid)
        note = muted_label('Skyfield works offline in a limited date range. JPL Horizons needs internet.')
        note.setWordWrap(True)
        sampling.addWidget(note)
        root.addLayout(sampling)
        graphs = QtWidgets.QVBoxLayout()
        graphs.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 8, 4)
        title = QtWidgets.QLabel('Points on graphs')
        title.setObjectName('groupTitle')
        head.addWidget(title)
        head.addStretch(1)
        self.horizon_title = muted_label('Horizon of')
        head.addWidget(self.horizon_title)
        graphs.addLayout(head)
        self.graph_layout = QtWidgets.QVBoxLayout()
        self.graph_layout.setSpacing(2)
        graphs.addLayout(self.graph_layout)
        self.graph_note = muted_label('')
        self.graph_note.setWordWrap(True)
        graphs.addWidget(self.graph_note)
        root.addLayout(graphs)
        root.addStretch(1)
        self.error = QtWidgets.QLabel()
        self.error.setObjectName('error')
        self.error.setWordWrap(True)
        root.addWidget(self.error)
        self.box = QtWidgets.QFrame()
        self.box.setObjectName('staleBox')
        self.box_layout = QtWidgets.QVBoxLayout(self.box)
        self.box_layout.setSpacing(10)
        self.banner = QtWidgets.QLabel()
        self.banner.setObjectName('warning')
        self.banner.setWordWrap(True)
        self.box_layout.addWidget(self.banner)
        self.apply_button = QtWidgets.QPushButton()
        self.apply_button.setObjectName('primary')
        self.apply_button.setFixedHeight(36)
        self.apply_button.clicked.connect(self.apply)
        self.box_layout.addWidget(self.apply_button)
        root.addWidget(self.box)
        self.load_saved()
        self.step.textChanged.connect(self.sync)
        self.unit.currentTextChanged.connect(self.sync)
        self.method.changed.connect(self.sync)

    def load_saved(self):
        saved = self.owner._saved_interval
        if saved:
            _, _, step, unit, method = saved
        else:
            step, unit, method = 60, 'minutes', self.owner._sky_method
        self.step.setText(str(step))
        self.unit.setCurrentText(unit)
        if method in self.method.buttons:
            self.method.set(method)

    def apply(self):
        if not self._has_analysis_points():
            self.error.setText('Select at least one point before starting an analysis.')
            return
        start, stop = (value.strftime(TIME_FORMAT) for value in self.shell.interval)
        try:
            self.owner.set_time_interval(start, stop, float(self.step.text()), self.unit.currentText(), self.method.get())
        except (ValueError, OverflowError) as error:
            self.error.setText(str(error))
        else:
            self.error.clear()
        self.shell.sync_state()

    def running(self):
        owner = self.owner
        return owner._graph_expanded and owner._compare_active and owner._compare is None and owner._enable_ephemeris

    def stale_reasons(self):
        used = self.owner._analysis_inputs
        if used is None or not self.shell.has_analysis():
            return []
        reasons = []
        if self.owner._analysis_points_override is None and self.owner.analysis_points() != used['points']:
            reasons.append('Points changed.')
        if self.shell.interval != (used['start'], used['stop']):
            reasons.append('Interval changed.')
        try:
            step = float(self.step.text()) * sky.TIME_UNITS[self.unit.currentText()]
        except ValueError:
            step = None
        if (step, self.method.get()) != (used['step'], used['method']):
            reasons.append('Sampling changed.')
        if terrain.EYE_HEIGHT_M != used['eye']:
            reasons.append('Observer height changed.')
        return reasons

    def analyze_state(self):
        if not self._has_analysis_points():
            return ('Update analysis' if self.shell.has_analysis() else 'Select points to analyze'), False
        if self.running():
            return 'Analyzing…', False
        if self.stale_reasons():
            return 'Update analysis', True
        count = len(self.owner.analysis_points())
        return (f'Analyze {counted(count, "point")}' if count else 'Analyze selected points'), True

    def _has_analysis_points(self):
        owner = self.owner
        return bool(owner.analysis_points())

    def sync(self, *_):
        available = self._has_analysis_points()
        for widget in (self.step, self.unit, self.method):
            widget.setEnabled(available)
            widget.setToolTip('' if available else 'Add and select at least one point to use analysis options.')
        reasons = self.stale_reasons()
        empty = bool(not available and self.shell.has_analysis())
        self.banner.setVisible(bool(reasons) or empty)
        if empty:
            self.banner.setText('Previous results are shown. No points are currently selected for a new analysis.')
        else:
            self.banner.setText('<b>Results are out of date.</b> ' + ' '.join(reasons))
        box_state = 'empty' if empty else bool(reasons)
        if self.box.property('stale') != box_state:
            self.box.setProperty('stale', box_state)
            self.box.style().unpolish(self.box)
            self.box.style().polish(self.box)
        margin = 12 if reasons or empty else 0
        self.box_layout.setContentsMargins(margin, margin, margin, margin)
        text, enabled = self.analyze_state()
        self.apply_button.setText(text)
        self.apply_button.setEnabled(enabled)
        self.sync_graph_points()
        if self.shell.points is not None:
            self.shell.points.sync_button()

    def sync_graph_points(self):
        owner = self.owner
        points = list(owner._compare_points) if owner._compare_active and owner._graph_expanded else []
        listed = len(points) > 1
        key = (tuple(points), tuple(owner._point_names.get(point) for point in points)) if listed else None
        if key != self.graph_key:
            self.graph_key = key
            self.clear_graph_rows()
            if listed:
                self.graph_rows = [GraphPointRow(self, index, point) for index, point in enumerate(points)]
                for row in self.graph_rows:
                    self.graph_layout.addWidget(row)
        self.horizon_title.setVisible(listed)
        self.graph_note.setVisible(not listed)
        if not listed:
            self.graph_note.setText('No points ticked: the analysis uses the selected location.' if not owner.analysis_points()
                                    else 'Run the analysis to choose which points are drawn.')
            return
        ready = owner._compare is not None
        for index, row in enumerate(self.graph_rows):
            row.sync(owner._compare_visible[index] if index < len(owner._compare_visible) else True,
                     index == owner._compare_index, ready)

    def clear_graph_rows(self):
        for row in self.graph_rows:
            self.graph_layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self.graph_rows = []

    def set_point_drawn(self, index, drawn):
        owner = self.owner
        if index < len(owner._compare_visible) and owner._compare_visible[index] != drawn:
            owner.toggle_point_visibility(index)

    def set_horizon(self, index):
        if self.owner._compare_index != index:
            self.owner.compare_activate(index)


class GraphPointRow(QtWidgets.QFrame):
    def __init__(self, panel, index, point):
        super().__init__()
        owner = panel.owner
        saved_index = owner.point_index(*point)
        color_index = index if saved_index is None else saved_index
        self.setObjectName('pointRow')
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(10)
        self.drawn = Check()
        self.drawn.setAccessibleName(f'Show point {index + 1} on graphs')
        self.drawn.toggled.connect(lambda value: panel.set_point_drawn(index, value))
        row.addWidget(self.drawn)
        badge = QtWidgets.QLabel(str(index + 1))
        badge.setObjectName('mono')
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(24, 24)
        badge.setStyleSheet(f'background: {point_color(color_index)}; color: #10151f; border-radius: 12px; font-weight: 500;')
        row.addWidget(badge)
        name = QtWidgets.QLabel(owner._point_names.get(point) or f'Point {index + 1}')
        name.setStyleSheet(f'font-weight: 500; color: {point_color(color_index)};')
        row.addWidget(name, 1)
        self.horizon = QtWidgets.QRadioButton()
        self.horizon.setAccessibleName(f'Show horizon of point {index + 1}')
        self.horizon.toggled.connect(lambda value: value and panel.set_horizon(index))
        row.addWidget(self.horizon)

    def sync(self, drawn, current, ready):
        state = (drawn, current, ready)
        if getattr(self, "_last_sync_state", None) == state:
            return
        self._last_sync_state = state
        for button, value in ((self.drawn, drawn), (self.horizon, current)):
            button.blockSignals(True)
            button.setChecked(value)
            button.blockSignals(False)
            button.setEnabled(ready)
        self.setProperty('on', current)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.horizon.isEnabled():
            self.horizon.setChecked(True)
            event.accept()
            return
        super().mousePressEvent(event)


class DataTable(QtWidgets.QTableWidget):
    def __init__(self, headers, rows, colors):
        super().__init__(len(rows), len(headers))
        self.headers, self.rows, self.colors = headers, rows, colors
        self.sort = None
        self.keys = [[cell_value(cell) for cell in row] for row in rows]
        for index, row in enumerate(self.keys):
            row[0] = index
        self.setObjectName('mono')
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(28)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self.sort_by)
        self.show_rows()

    def sort_by(self, column):
        self.sort = compare.next_sort(self.sort, column)
        self.show_rows()

    def show_rows(self):
        order = compare.sort_order(self.keys, self.sort)
        labels = list(self.headers)
        if self.sort is not None:
            labels[self.sort[0]] += ' ▼' if self.sort[1] else ' ▲'
        self.setHorizontalHeaderLabels(labels)
        for column in range(1, len(labels)):
            self.horizontalHeaderItem(column).setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for position, index in enumerate(order):
            for column, text in enumerate(self.rows[index]):
                item = QtWidgets.QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setIcon(dot_icon(self.colors[index]))
                    item.setForeground(QtGui.QBrush(self.colors[index]))
                self.setItem(position, column, item)
        self.resizeColumnsToContents()

    def selected_text(self):
        cells = {(item.row(), item.column()) for item in self.selectedIndexes()}
        rows = sorted({row for row, _ in cells})
        columns = sorted({column for _, column in cells})
        return '\n'.join('\t'.join(self.item(row, column).text() if (row, column) in cells else '' for column in columns)
                         for row in rows)

    def keyPressEvent(self, event):
        if event.matches(QtGui.QKeySequence.Copy):
            QtWidgets.QApplication.clipboard().setText(self.selected_text())
        else:
            super().keyPressEvent(event)


class TablesPanel(QtWidgets.QFrame):
    def __init__(self, shell, owner):
        super().__init__()
        self.setObjectName('tables')
        self.shell, self.owner = shell, owner
        self.source = self.key = None
        self.data = []
        self.shown = False
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 10, 16, 8)
        root.setSpacing(8)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(muted_label('Direction averages'))
        self.mode = Segmented((ABOVE_ONLY, ALL_SAMPLES))
        self.mode.setFixedWidth(250)
        self.mode.changed.connect(self._mode_changed)
        top.addWidget(self.mode)
        top.addStretch(1)
        root.addLayout(top)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)
        self.notes_toggle = QtWidgets.QPushButton('› Table notes')
        self.notes_toggle.setObjectName('ghost')
        self.notes_toggle.setCheckable(True)
        self.notes_toggle.setFixedHeight(26)
        self.notes_toggle.toggled.connect(self._notes_toggled)
        root.addWidget(self.notes_toggle, 0, Qt.AlignLeft)
        self.notes = muted_label('')
        self.notes.setWordWrap(True)
        self.notes_scroll = QtWidgets.QScrollArea()
        self.notes_scroll.setWidgetResizable(True)
        self.notes_scroll.setMaximumHeight(110)
        self.notes_scroll.setWidget(self.notes)
        self.notes_scroll.hide()
        root.addWidget(self.notes_scroll)

    def _mode_changed(self, value):
        if (value == ABOVE_ONLY) != self.owner._stats_above_only:
            self.owner.toggle_stats_mode()

    def _notes_toggled(self, on):
        self.notes_toggle.setText(('⌄' if on else '›') + ' Table notes')
        self.notes_scroll.setVisible(on)

    def _open_height(self):
        split = self.shell.right
        sizes = split.sizes()
        total = sum(sizes)
        sizes[2] = TABLES_HEIGHT
        sizes[0] = max(1, total - sizes[1] - TABLES_HEIGHT)
        split.setSizes(sizes)

    def sync(self):
        owner = self.owner
        result = owner._compare
        self.shown = bool(not self.shell._zen and owner._graph_expanded and owner._compare_active and result is not None
                          and result.get('stats') is not None and len(result['points']) > 1)
        if not self.shown:
            self.hide()
            return
        self.mode.set(ABOVE_ONLY if owner._stats_above_only else ALL_SAMPLES)
        names = tuple(owner._point_names.get(point) for point in result['points'])
        if result['stats'] is not self.source or (names, owner._stats_above_only) != self.key:
            self.source, self.key = result['stats'], (names, owner._stats_above_only)
            self.rebuild(result)
        if self.isHidden():
            self.show()
            QtCore.QTimer.singleShot(0, self._open_height)

    def rebuild(self, result):
        owner = self.owner
        sorts = [self.tabs.widget(index).sort for index in range(self.tabs.count())]
        current = self.tabs.currentIndex()
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        colors = [QtGui.QColor.fromRgbF(*owner.compare_color(index))
                  for index in range(len(result['points']))]
        self.data = []
        for number, (title, header, lines, _, spans) in enumerate(compare.build_tables(result['points'], result['stats'])):
            headers = ['\n'.join(part for part in (header[0][a:b].strip(), header[1][a:b].strip()) if part) for a, b in spans]
            rows = [[line[a:b].strip() for a, b in spans] for line in lines]
            for index, (point, row) in enumerate(zip(result['points'], rows)):
                row[0] = f'{index + 1} {owner._point_names.get(point) or f"Point {index + 1}"}'
            table = DataTable(headers, rows, colors)
            if number < len(sorts):
                table.sort = sorts[number]
                table.show_rows()
            self.tabs.addTab(table, title)
            self.data.append((title, headers, rows))
        if 0 <= current < self.tabs.count():
            self.tabs.setCurrentIndex(current)
        self.notes.setText(compare.table_notes(result['stats_span'], owner._stats_above_only))

    def export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export CSV', 'moon_analysis.csv', 'CSV files (*.csv)')
        if path:
            if not path.lower().endswith('.csv'):
                path += '.csv'
            self.export_to(path)

    def export_to(self, path):
        try:
            with open(path, 'w', newline='', encoding='utf-8') as file:
                out = csv.writer(file)
                for index, (title, headers, rows) in enumerate(self.data):
                    if index:
                        out.writerow([])
                    out.writerow([title])
                    out.writerow([header.replace('\n', ' ') for header in headers])
                    out.writerows(rows)
        except OSError as error:
            QtWidgets.QMessageBox.warning(self, 'Export CSV', f'Could not save the file: {error}')


class TopBar(QtWidgets.QFrame):
    def __init__(self, shell):
        super().__init__()
        self.shell = shell
        self.setObjectName('topBar')
        self.setFixedHeight(48)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(16, 0, 16, 0)
        row.setSpacing(12)
        self.tools = QtWidgets.QWidget()
        tools = QtWidgets.QHBoxLayout(self.tools); tools.setContentsMargins(0, 0, 0, 0); tools.setSpacing(4)
        self.tool_buttons = {}
        tool_group = QtWidgets.QButtonGroup(self.tools)
        for name, icon, tip in (('Navigate', 'hand', 'Navigate: rotate and select existing points'),
                                ('Add', 'target', 'Add: click to select, Shift-click to add; drag to orbit')):
            button = QtWidgets.QToolButton(); button.setIcon(line_icon(icon)); button.setIconSize(QtCore.QSize(18, 18))
            button.setCheckable(True); button.setToolTip(tip); button.setAccessibleName(name + ' tool')
            button.clicked.connect(lambda _checked=False, n=name:self._tool_changed(n))
            tool_group.addButton(button); tools.addWidget(button); self.tool_buttons[name] = button
        self.tool_buttons['Navigate'].setChecked(True)
        row.addWidget(self.tools)
        self.summary = muted_label('')
        row.addWidget(self.summary)
        row.addStretch(1)
        self.export = QtWidgets.QPushButton('Export CSV')
        self.export.clicked.connect(lambda: shell.tables.export())
        row.addWidget(self.export)

    def _tool_changed(self, value):
        if self.shell.owner._area_drawing:
            self.shell.cancel_area_tool()
        self.shell.owner._map_tool = value
        self.shell.vtk_widget.setFocus()

    def _populate_layers(self):
        self.shell.layer_menu.clear()
        owner = self.shell.owner
        for mode, label in (('Color', 'Color'), ('Elevation', 'Terrain'), ('Slope', 'Slope')):
            action = self.shell.layer_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(owner._map_wanted == mode)
            action.triggered.connect(lambda _checked=False, m=mode: self.shell.select_map(m))

    def sync(self):
        shell = self.shell
        analysis = shell.has_analysis()
        self.tools.setVisible(not shell.owner._graph_expanded)
        shell.coord_field.setVisible(not shell.owner._graph_expanded)
        shell.layer_button.setVisible(not shell.owner._graph_expanded)
        self.summary.setText(shell.analysis_summary() if analysis and shell.owner._graph_expanded else '')
        self.export.setEnabled(shell.tables.shown)
        self.export.setToolTip('' if shell.tables.shown else 'Tables appear when two or more points are analyzed.')


class AnalysisHeader(QtWidgets.QFrame):
    """Horizon graph header, positioned within the scrolling viewport."""
    def __init__(self, shell, parent=None):
        super().__init__(parent)
        self.shell = shell
        self.setObjectName('analysisHeader')
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(6)
        values = QtWidgets.QHBoxLayout()
        values.setSpacing(14)
        self.earth = QtWidgets.QLabel('Earth: --')
        self.sun = QtWidgets.QLabel('Sun: --')
        self.horizon = muted_label('Horizon: --')
        self.scale = muted_label('Scale: --')
        for widget in (self.earth, self.sun, self.horizon, self.scale):
            values.addWidget(widget)
        values.addStretch(1)
        root.addLayout(values)
        controls = QtWidgets.QHBoxLayout()
        self.body_checks = {}
        for name in ('Sun', 'Earth'):
            check = Check('Show ' + name)
            check.toggled.connect(lambda checked, body=name: self._toggle_body(body, checked))
            controls.addWidget(check)
            self.body_checks[name] = check
        self.layer = QtWidgets.QToolButton()
        self.layer.setIcon(line_icon('layers'))
        self.layer.setIconSize(QtCore.QSize(18, 18))
        self.layer.setToolTip('Choose horizon layer')
        self.layer.setAccessibleName('Choose horizon layer')
        self.layer.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.layer_menu = QtWidgets.QMenu(self.layer)
        self.layer_menu.aboutToShow.connect(self._populate_layers)
        self.layer.setMenu(self.layer_menu)
        controls.addWidget(self.layer)
        controls.addStretch(1)
        self.timestamp = QtWidgets.QLabel('')
        self.timestamp.setObjectName('mono')
        controls.addWidget(self.timestamp)
        self.previous = QtWidgets.QPushButton('< Prev')
        self.next = QtWidgets.QPushButton('Next >')
        self.previous.clicked.connect(lambda: self.shell.owner.step_time(-1))
        self.next.clicked.connect(lambda: self.shell.owner.step_time(1))
        controls.addWidget(self.previous)
        controls.addWidget(self.next)
        root.addLayout(controls)
        self.hide()

    def _toggle_body(self, body, checked):
        owner = self.shell.owner
        if owner is not None and owner._horizon_bodies[body] != checked:
            owner._horizon_bodies[body] = checked
            owner.update_profile(owner.horizon_angles)
            owner.window.Render()

    def _populate_layers(self):
        owner = self.shell.owner
        self.layer_menu.clear()
        if owner is None:
            return
        for mode, label in (('Color', 'Color'), ('Elevation', 'Terrain'), ('Slope', 'Slope')):
            action = self.layer_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(owner._horizon_wanted == mode)
            action.triggered.connect(lambda _checked=False, value=mode: self._set_layer(value))

    def _set_layer(self, mode):
        owner = self.shell.owner
        owner._horizon_wanted = mode
        owner._apply_loaded_maps()
        owner._schedule_map_generation()
        owner.window.Render()

    def sync(self):
        owner = self.shell.owner
        visible = bool(owner and owner._graph_expanded and not self.shell._zen)
        self.setVisible(visible)
        if not visible:
            return
        source = sky.SKY_METHODS[owner._sky_method][0]
        for body, check in self.body_checks.items():
            check.blockSignals(True)
            check.setChecked(owner._horizon_bodies[body])
            check.blockSignals(False)
        self.layer.setToolTip(f'Horizon layer: {owner._horizon_wanted}')
        index = min(owner._time_index, len(owner._epochs) - 1)
        self.timestamp.setText(f'{owner._epochs[index]:%Y-%m-%d %H:%M:%S} UTC  ({index + 1} / {len(owner._epochs)})')
        self.previous.setEnabled(index > 0)
        self.next.setEnabled(index + 1 < len(owner._epochs))
        for body_name, label, color in (('Earth', self.earth, '#2dc976'), ('Sun', self.sun, '#fa801f')):
            body = (owner._sky_data or {}).get(f'{body_name} {source}')
            value = '--'
            if body is not None and len(body['el']) > index:
                elevation = float(body['el'][index])
                clearance = elevation - float(body['terrain'][index])
                value = f'{elevation:+.1f}°'
            label.setText(f'<span style="color:{color}">■</span> {body_name}: <span style="color:{color}">{value}</span>')
        angles = owner.horizon_angles
        if angles is not None and len(angles):
            self.horizon.setText(f'Horizon: {min(angles):+.1f}° to {max(angles):+.1f}°')
        self.scale.setText(f'Scale: {getattr(owner, "_profile_low", 0):+.1f}° to {getattr(owner, "_profile_high", 0):+.1f}°')


class GraphHeader(QtWidgets.QFrame):
    """A graph-specific header sharing the horizon header's Qt styling."""
    TITLES = {'Sun': 'Sun height above terrain vs time (UTC)',
              'Earth': 'Earth height above terrain vs time (UTC)',
              'Lunar': 'Sun visible % per lunar day vs time (UTC)'}

    def __init__(self, shell, key, parent):
        super().__init__(parent)
        self.shell, self.key = shell, key
        self.setObjectName('analysisHeader')
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(6)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(9)
        self.marker = QtWidgets.QLabel('■')
        self.title = QtWidgets.QLabel(self.TITLES[key])
        self.title.setObjectName('mono')
        self.count = QtWidgets.QLabel('')
        self.mean = QtWidgets.QLabel('')
        self.mean.setStyleSheet('color: #32cbdf;')
        for widget in (self.marker, self.title, self.count, self.mean):
            top.addWidget(widget)
        top.addStretch(1)
        root.addLayout(top)
        row = QtWidgets.QHBoxLayout()
        self.current = QtWidgets.QLabel('')
        self.current.setObjectName('mono')
        row.addWidget(self.current)
        row.addStretch(1)
        root.addLayout(row)
        self._mean_key = None
        self.hide()

    def sync(self):
        owner = self.shell.owner
        active = bool(owner and owner._graph_expanded and self.key in owner.panel_keys() and not self.shell._zen)
        if not active:
            self.hide()
            return
        index = owner.panel_keys().index(self.key) + 1
        viewport = self.parentWidget()
        width, height = viewport.width(), viewport.height()
        y = round((1.0 + index * hud.PANEL_HEIGHT - owner._scroll - .96) * height)
        x, card_width = round(.025 * width), round(.945 * width)
        header_height = self.sizeHint().height()
        self.setGeometry(x, y, max(1, card_width), header_height)
        if y >= height or y + header_height <= 0:
            self.hide()
            return
        self.show()
        self.raise_()
        point_count = owner.compare_n() if owner._compare_active else 1
        self.count.setText(f'{point_count} point' + ('' if point_count == 1 else 's'))
        point_index = owner._compare_index if owner._compare_active else 0
        rgb = owner.compare_color(point_index)
        color = '#%02x%02x%02x' % tuple(round(component * 255) for component in rgb)
        if getattr(self, '_last_color', None) != color:
            self._last_color = color
            self.marker.setStyleSheet(f'color: {color};')
            self.mean.setStyleSheet(f'color: {color};')
        series = owner.selected_graph_series(self.key)
        sample = min(owner._time_index, len(owner._epochs) - 1)
        meta = f"UTC: {owner._epochs[sample]:%m-%d %H:%M}  |  Sample {sample + 1} / {len(owner._epochs)}"
        if series is not None and len(series) > sample and np.isfinite(series[sample]):
            unit = '%' if self.key == 'Lunar' else '°'
            label = 'Visibility' if self.key == 'Lunar' else 'Height'
            meta += f'  |  <span style="color:{color}">{label}: {series[sample]:.1f}{unit}</span>'
        self.current.setText(meta)
        token = (id(owner._compare), point_index, id(owner._panel_state), self.key)
        if token != self._mean_key:
            self._mean_key = token
            if series is not None:
                array = np.asarray(series, dtype=float)
                finite = np.isfinite(array)
                count = int(np.count_nonzero(finite))
                total = float(np.sum(array, where=finite, initial=0))
            else:
                count, total = 0, 0.0
            unit = '%' if self.key == 'Lunar' else '°'
            self.mean.setText(f'{total/count:.1f}{unit} Mean' if count else '')


class AnalysisTimeBar(QtWidgets.QFrame):
    """Qt-owned sample control, fixed outside the scrolling VTK graphs."""
    def __init__(self, shell):
        super().__init__()
        self.shell = shell
        self.setObjectName('timeline')
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 4, 14, 4)
        self.start = muted_label('')
        self.current = QtWidgets.QLabel('')
        self.current.setObjectName('mono')
        self.end = muted_label('')
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setAccessibleName('Analysis time')
        self.slider.valueChanged.connect(self._changed)
        self.current.hide()
        for widget in (self.start, self.slider, self.end):
            layout.addWidget(widget, 1 if widget is self.slider else 0)
        self.hide()

    def _changed(self, value):
        if self.shell.owner is not None:
            self.shell.owner.set_time_index(value)

    def sync(self):
        owner = self.shell.owner
        visible = bool(owner and owner._graph_expanded and not self.shell._zen and owner._epochs)
        self.setVisible(visible)
        if not visible:
            return
        epochs = owner._epochs
        index = min(owner._time_index, len(epochs)-1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(epochs)-1))
        self.slider.setValue(index)
        self.slider.setEnabled(len(epochs) > 1)
        self.slider.blockSignals(False)
        fmt = '%m-%d %H:%M'
        self.start.setText(epochs[0].strftime(fmt))
        self.current.setText(epochs[index].strftime(fmt))
        self.end.setText(epochs[-1].strftime(fmt))


class CoordinateField(QtWidgets.QLineEdit):
    escaped = QtCore.Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.escaped.emit()
            return
        super().keyPressEvent(event)


HELP_SECTIONS = (
    ('Mouse', (('Navigate: drag', 'Orbit'), ('Middle / Shift-drag', 'Pan'), ('Ctrl-drag', 'Orbit around the cursor point'),
               ('Wheel', 'Zoom'), ('Add: click', 'Choose a location'), ('Add: Shift-click', 'Add a temporary selection'),
               ('Add: drag', 'Orbit'), ('Search area: drag', 'Set search bounds; right-click cancels'))),
    ('Navigation', (('G', 'Go to coordinates'), ('V', 'Terrain view'), ('R', 'Reset view'), ('F', 'Fullscreen'),
                    ('Esc', 'Exit zen mode'))),
    ('Search and analysis', (('S', 'Search tab'), ('T', 'Analysis tab'), ('I', 'Analyze selected result'),
                             ('P', 'Manage points'), ('[ / ]', 'Previous / next time step'))),
    ('Display', (('H', 'View menu'), ('M', 'Change map'), ('K', 'Map legend'), ('L', 'Coordinate grid'),
                 ('E', 'Observer height'))),
)


def line_icon(kind):
    pixmap = QtGui.QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtGui.QPen(QtGui.QColor('#9fb0c3'), 3, Qt.SolidLine, Qt.RoundCap))
    if kind == 'target':
        painter.drawEllipse(QtCore.QPointF(16, 16), 10, 10)
        for x1, y1, x2, y2 in ((16, 2, 16, 8), (16, 24, 16, 30), (2, 16, 8, 16), (24, 16, 30, 16)):
            painter.drawLine(x1, y1, x2, y2)
    elif kind == 'layers':
        for y in (8, 15, 22):
            painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(4, y), QtCore.QPointF(16, y + 6),
                                                 QtCore.QPointF(28, y), QtCore.QPointF(16, y - 6),
                                                 QtCore.QPointF(4, y)]))
    elif kind == 'area':
        pen = painter.pen(); pen.setStyle(Qt.DashLine); painter.setPen(pen)
        painter.drawRect(5, 6, 22, 20)
    elif kind == 'hand':
        path = QtGui.QPainterPath(QtCore.QPointF(9, 16))
        path.lineTo(9, 5); path.cubicTo(9, 1, 14, 1, 14, 5); path.lineTo(14, 12)
        path.lineTo(16, 6); path.cubicTo(17, 3, 21, 5, 20, 8); path.lineTo(19, 13)
        path.lineTo(22, 9); path.cubicTo(24, 7, 27, 10, 25, 12); path.lineTo(23, 17)
        path.cubicTo(27, 15, 29, 19, 26, 22); path.lineTo(21, 28); path.lineTo(12, 28)
        path.lineTo(5, 20); path.cubicTo(3, 17, 6, 14, 9, 16); painter.drawPath(path)
    else:
        painter.drawLine(8, 8, 24, 24)
        painter.drawLine(24, 8, 8, 24)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QtGui.QIcon(pixmap)


def icon_button(kind, tooltip, callback):
    button = QtWidgets.QToolButton()
    button.setObjectName('icon')
    button.setIcon(line_icon(kind))
    button.setIconSize(QtCore.QSize(16, 16))
    button.setFixedSize(30, 30)
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip)
    button.clicked.connect(callback)
    return button


def point_color(index):
    r, g, b = (round(255 * value) for value in compare.POINT_COLORS[index % len(compare.POINT_COLORS)])
    return f'rgb({r},{g},{b})'


class PointRow(QtWidgets.QFrame):
    def __init__(self, panel, index, point):
        super().__init__()
        self.panel, self.index, self.point = panel, index, point
        owner = panel.owner
        self.setObjectName('pointRow')
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(10)
        included = point not in owner._point_off
        self.check = Check()
        self.check.setChecked(included)
        self.check.setToolTip('Take part in the analysis')
        self.check.setAccessibleName(f'Include point {index + 1} in analysis')
        self.check.toggled.connect(lambda value: owner.set_point_included(point, value))
        row.addWidget(self.check)
        badge = QtWidgets.QLabel(str(index + 1))
        badge.setObjectName('mono')
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(24, 24)
        badge.setStyleSheet(f'background: {point_color(index)}; color: #10151f; border-radius: 12px; font-weight: 500;')
        row.addWidget(badge)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        self.name = QtWidgets.QLabel(owner._point_names.get(point) or f'Point {index + 1}')
        self.name.setStyleSheet(f'font-weight: 500; color: {point_color(index)};')
        self.name.setToolTip('Double-click to rename')
        self.name.installEventFilter(self)
        self.editor = QtWidgets.QLineEdit()
        self.editor.setPlaceholderText(f'Point {index + 1}')
        self.editor.setFixedHeight(26)
        self.editor.setStyleSheet('min-height: 26px; padding: 0 8px;')
        self.editor.hide()
        self.editor.editingFinished.connect(self.finish_rename)
        coords = QtWidgets.QLabel(f'{point[0]:.2f}, {point[1]:.2f}')
        coords.setObjectName('muted')
        coords.setStyleSheet(f'font-family: "{MONO_FAMILY}";')
        text.addWidget(self.name)
        text.addWidget(self.editor)
        text.addWidget(coords)
        row.addLayout(text, 1)
        row.addWidget(icon_button('cross', f'Remove point {index + 1}', lambda: panel.remove(point)))

    def eventFilter(self, obj, event):
        if obj is self.name and event.type() == QtCore.QEvent.MouseButtonDblClick:
            self.start_rename()
            return True
        return super().eventFilter(obj, event)

    def start_rename(self):
        self.editor.setText(self.panel.owner._point_names.get(self.point, ''))
        self.name.hide()
        self.editor.show()
        self.editor.setFocus()
        self.editor.selectAll()

    def finish_rename(self):
        if not self.editor.isVisible():
            return
        owner = self.panel.owner
        owner.set_point_name(self.point, self.editor.text())
        self.name.setText(owner._point_names.get(self.point) or f'Point {self.index + 1}')
        self.editor.hide()
        self.name.show()
        self.panel.shell.sync_state()

    def mousePressEvent(self, event):
        self.panel.go_to(self.point)


class PointsPanel(QtWidgets.QWidget):
    def __init__(self, shell, owner):
        super().__init__()
        self.shell, self.owner = shell, owner
        self.rows = []
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 12, 8, 12)
        root.setSpacing(4)
        hint = QtWidgets.QLabel('Ticked points take part in the analysis.')
        hint.setObjectName('muted')
        hint.setContentsMargins(12, 0, 12, 6)
        root.addWidget(hint)
        self.empty = QtWidgets.QLabel('Use the Add tool to choose locations, then press Add to analysis list, or add results from a search.')
        self.empty.setObjectName('muted')
        self.empty.setWordWrap(True)
        self.empty.setContentsMargins(12, 10, 12, 10)
        self.empty.setAlignment(Qt.AlignTop)
        root.addWidget(self.empty, 1)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.list_widget = QtWidgets.QWidget()
        self.list_layout = QtWidgets.QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(4)
        self.list_layout.addStretch(1)
        self.scroll.setWidget(self.list_widget)
        root.addWidget(self.scroll, 1)
        self.error = QtWidgets.QLabel()
        self.error.setObjectName('error')
        self.error.setWordWrap(True)
        self.error.setContentsMargins(12, 0, 12, 0)
        root.addWidget(self.error)
        bar = QtWidgets.QFrame()
        bar.setObjectName('barTop')
        bottom = QtWidgets.QVBoxLayout(bar)
        bottom.setContentsMargins(0, 12, 0, 4)
        bottom.setSpacing(10)
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        self.count = QtWidgets.QLabel()
        self.count.setObjectName('muted')
        controls.addWidget(self.count, 1)
        self.load_button = QtWidgets.QPushButton('Load…')
        menu = QtWidgets.QMenu(self.load_button)
        menu.addAction('Replace current points').triggered.connect(lambda: self.load(False))
        menu.addAction('Add to analysis list').triggered.connect(lambda: self.load(True))
        self.load_button.setMenu(menu)
        controls.addWidget(self.load_button)
        self.save_button = QtWidgets.QPushButton('Save…')
        self.save_button.clicked.connect(self.save)
        controls.addWidget(self.save_button)
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.clear_button.setObjectName('ghost')
        self.clear_button.clicked.connect(lambda: owner.set_points([]))
        controls.addWidget(self.clear_button)
        bottom.addLayout(controls)
        self.analyze_button = QtWidgets.QPushButton()
        self.analyze_button.setObjectName('primary')
        self.analyze_button.setFixedHeight(36)
        self.analyze_button.clicked.connect(self.analyze)
        bottom.addWidget(self.analyze_button)
        root.addWidget(bar)
        self.refresh()

    def refresh(self):
        owner = self.owner
        for row in self.rows:
            self.list_layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self.rows = [PointRow(self, index, point) for index, point in enumerate(owner._points)]
        for index, row in enumerate(self.rows):
            self.list_layout.insertWidget(index, row)
        total = len(owner._points)
        ticked = len(owner.analysis_points())
        self.empty.setVisible(not total)
        self.scroll.setVisible(bool(total))
        self.error.clear()
        self.count.setText(f'{total} of {compare.MAX_POINTS} points')
        self.save_button.setEnabled(bool(total))
        self.clear_button.setEnabled(bool(total))
        self.analyze_button.setToolTip('' if ticked else 'Tick at least one point to analyze.')
        self.sync_button()
        self.mark_selected()

    def sync_button(self):
        text, enabled = self.shell.time.analyze_state()
        ticked = len(self.owner.analysis_points())
        self.analyze_button.setText(text if ticked else 'Analyze 0 points')
        self.analyze_button.setEnabled(enabled and bool(ticked))

    def mark_selected(self):
        selected = self.owner.point_index(self.owner.lat, self.owner.lon)
        for index, row in enumerate(self.rows):
            row.setProperty('on', index == selected)
            row.style().unpolish(row)
            row.style().polish(row)

    def go_to(self, point):
        self.shell.go_to(f'{point[0]!r}, {point[1]!r}')

    def remove(self, point):
        self.owner.set_points([other for other in self.owner._points if other != point])

    def analyze(self):
        self.shell.time.apply()
        if not self.shell.time.error.text():
            self.shell.show_panel('time')

    def load(self, append):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load points', '', 'Points (*.json *.txt);;All files (*)')
        if path:
            self.load_file(path, append)

    def load_file(self, path, append):
        owner = self.owner
        try:
            points, names = compare.load_named_points(path)
            if append:
                points = [point for point in points if not any(compare.same_point(point, other) for other in owner._points)]
                names = {point: name for point, name in names.items() if point in points}
                if len(owner._points) + len(points) > compare.MAX_POINTS:
                    raise ValueError(f'A set holds at most {compare.MAX_POINTS} points.')
                names = {**owner._point_names, **names}
                points = owner._points + points
        except ValueError as error:
            self.error.setText(str(error))
            return
        owner.set_points(points, names)

    def save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save points', 'moon_points.json', 'JSON (*.json)')
        if path:
            self.save_file(path)

    def save_file(self, path):
        try:
            compare.save_points(path, self.owner._points, self.owner._point_names)
        except OSError as error:
            self.error.setText(f'Could not save the file: {error}')


class LocationCard(QtWidgets.QFrame):
    def __init__(self, shell, parent):
        super().__init__(parent)
        self.shell = shell
        self.key = None
        self.setObjectName('card')
        self.setFixedWidth(236)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet('font-weight: 600;')
        self.title.setWordWrap(True)
        head.addWidget(self.title, 1)
        self.tag = QtWidgets.QLabel('not saved')
        self.tag.setObjectName('muted')
        head.addWidget(self.tag, 0, Qt.AlignTop)
        layout.addLayout(head)
        self.coords = QtWidgets.QLabel()
        self.coords.setStyleSheet(f'font-family: "{MONO_FAMILY}"; font-size: 15px;')
        layout.addWidget(self.coords)
        self.selected_count = muted_label('')
        layout.addWidget(self.selected_count)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        self.values = {}
        for row, name in enumerate(('Elevation', 'Slope', 'Horizon')):
            label = QtWidgets.QLabel(name)
            label.setObjectName('muted')
            value = QtWidgets.QLabel()
            value.setStyleSheet(f'font-family: "{MONO_FAMILY}";')
            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1, Qt.AlignRight)
            self.values[name] = value
        layout.addLayout(grid)
        self.button = QtWidgets.QPushButton()
        self.button.clicked.connect(self.act)
        layout.addWidget(self.button)

    def act(self):
        owner = self.shell.owner
        unsaved = [point for point in owner._selected_locations if owner.point_index(*point) is None]
        if unsaved:
            owner.add_current_point()
            if all(owner.point_index(*point) is not None for point in unsaved):
                self.shell.show_panel('points')
        else:
            owner.toggle_terrain_view()

    def update_state(self):
        owner = self.shell.owner
        visible = not (owner._graph_expanded or owner._terrain_view or self.shell._zen or getattr(owner, '_observer_suppressed', False))
        if visible:
            index = owner.point_index(owner.lat, owner.lon)
            full = len(owner._points) >= compare.MAX_POINTS
            key = (owner.lat, owner.lon, index, owner._point_names.get(owner._points[index]) if index is not None else None,
                   full, owner._selected_stats, tuple(owner._selected_locations), tuple(owner._points))
            if key != self.key:
                self.key = key
                self.refill(index, full)
        if visible != self.isVisible():
            self.setVisible(visible)
        if visible:
            self.reposition()

    def refill(self, index, full):
        owner = self.shell.owner
        selected = owner._selected_locations or [(owner.lat, owner.lon)]
        unsaved = [point for point in selected if owner.point_index(*point) is None]
        elevation, slope, low, high = owner._selected_stats
        if index is None:
            self.title.setText('Selected location')
            self.button.setText('Add to analysis list')
            self.button.setObjectName('primary')
            self.button.setEnabled(not full)
            self.button.setToolTip(f'A set holds at most {compare.MAX_POINTS} points.' if full else '')
        else:
            name = owner._point_names.get(owner._points[index])
            self.title.setText(f'Point {index + 1} · {name}' if name else f'Point {index + 1}')
            self.button.setText('Terrain view')
            self.button.setObjectName('')
            self.button.setEnabled(True)
            self.button.setToolTip('')
        self.title.setStyleSheet('font-weight: 600;' + (f' color: {point_color(index)};' if index is not None else ''))
        if unsaved:
            self.button.setText('Add to analysis list' if len(unsaved) == 1 else f'Add {len(unsaved)} to analysis list')
            self.button.setObjectName('primary')
            available = len(owner._points) + len(unsaved) <= compare.MAX_POINTS
            self.button.setEnabled(available)
            self.button.setToolTip('Add the selected locations to the analysis list.' if available
                                   else f'The saved list holds at most {compare.MAX_POINTS} points.')
        self.selected_count.setText(f'Selected: {len(selected)} · Saved: {len(owner._points)}/{compare.MAX_POINTS}')
        self.tag.setVisible(index is None)
        self.button.style().unpolish(self.button)
        self.button.style().polish(self.button)
        self.coords.setText(f'{owner.lat:.2f}°, {owner.lon:.2f}°')
        self.values['Elevation'].setText(f'{elevation:+,.0f} m'.replace(',', ' '))
        self.values['Slope'].setText(f'{slope:.1f}°')
        self.values['Horizon'].setText(f'{low:+.2f}° … {high:+.2f}°')
        self.adjustSize()

    def reposition(self):
        self.move(max(0, self.parentWidget().width() - self.width() - 16), 16)
        self.raise_()


class MainShell(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Moonchaser')
        self.resize(1440, 900)
        self.owner = None
        self.search = None
        self.time = None
        self.tables = None
        self.points = None
        self.card = None
        self.current_panel = None
        self.interval = None
        self.help_dialog = None
        self.pages = {}
        self.tab_keys = []
        self._syncing = False
        self._shown_location = None
        self._zen = False
        self._zen_sidebar = False
        self._sidebar_width = 360
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = self._build_header()
        self.header.installEventFilter(self)
        outer.addWidget(self.header)
        self.zen_hint = QtWidgets.QPushButton('Exit zen mode  ·  Esc')
        self.zen_hint.clicked.connect(self.toggle_zen)
        self.zen_hint.hide()
        outer.addWidget(self.zen_hint, 0, Qt.AlignRight)
        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(3)
        self.sidebar = self._build_sidebar()
        self.right = QtWidgets.QSplitter(Qt.Vertical)
        self.right.setChildrenCollapsible(False)
        self.right.setHandleWidth(3)
        self.graph_container = QtWidgets.QWidget(self.right)
        graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)
        self.vtk_widget = GlobeWidget(self.graph_container)
        self.timebar = AnalysisTimeBar(self)
        graph_layout.addWidget(self.vtk_widget, 1)
        graph_layout.addWidget(self.timebar)
        self.right.addWidget(self.graph_container)
        self.right.setStretchFactor(0, 1)
        self.topbar = TopBar(self)
        self.analysis_header = AnalysisHeader(self, self.vtk_widget)
        self.graph_headers = {key: GraphHeader(self, key, self.vtk_widget) for key in ('Sun', 'Earth', 'Lunar')}
        column = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(column)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        stack.addWidget(self.topbar)
        stack.addWidget(self.right, 1)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(column)
        self.card = LocationCard(self, self.vtk_widget)
        self.card.hide()
        self.vtk_widget.installEventFilter(self)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([self._sidebar_width, 1080])
        outer.addWidget(self.splitter, 1)
        self.interval_popup = self._build_interval_popup()
        shortcut_type = getattr(QtGui, 'QShortcut', None) or QtWidgets.QShortcut
        self.zen_shortcut = shortcut_type(QtGui.QKeySequence('Escape'), self)
        self.zen_shortcut.activated.connect(self.exit_zen)

    def _build_header(self):
        header = QtWidgets.QFrame()
        header.setObjectName('header')
        header.setFixedHeight(48)
        bar = QtWidgets.QHBoxLayout(header)
        bar.setContentsMargins(14, 0, 14, 0)
        bar.setSpacing(12)
        self.sidebar_toggle = QtWidgets.QPushButton('☰')
        self.sidebar_toggle.setToolTip('Show or hide the sidebar')
        self.sidebar_toggle.setAccessibleName('Toggle sidebar')
        self.sidebar_toggle.setFixedWidth(40)
        self.sidebar_toggle.clicked.connect(self.toggle_sidebar)
        bar.addWidget(self.sidebar_toggle)
        self.coord_field = CoordinateField(header)
        self.coord_field.setObjectName('pointSearch')
        self.coord_field.setPlaceholderText('Find point by lat, lon')
        self.coord_field.setAccessibleName('Go to latitude, longitude')
        self.coord_field.setFixedSize(190, 32)
        self.coord_field.returnPressed.connect(self.go_to_coordinates)
        self.coord_field.textEdited.connect(lambda _text: self.set_coordinate_error(''))
        self.coord_field.escaped.connect(lambda: self.vtk_widget.setFocus())
        self.interval_button = QtWidgets.QPushButton()
        self.interval_button.setObjectName('mono')
        self.interval_button.setToolTip('Time interval used by search and analysis')
        self.interval_button.clicked.connect(self.show_interval_popup)
        bar.addWidget(self.interval_button)
        bar.addStretch(1)
        bar.addSpacing(self.coord_field.width())
        bar.addStretch(1)
        self.layer_button = QtWidgets.QToolButton()
        self.layer_button.setIcon(line_icon('layers'))
        self.layer_button.setIconSize(QtCore.QSize(18, 18))
        self.layer_button.setToolTip('Choose map layer')
        self.layer_button.setAccessibleName('Choose map layer')
        self.layer_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.layer_menu = QtWidgets.QMenu(self.layer_button)
        self.layer_menu.aboutToShow.connect(lambda: self.topbar._populate_layers())
        self.layer_button.setMenu(self.layer_menu)
        bar.addWidget(self.layer_button)
        self.view_button = QtWidgets.QPushButton('HUD (H)')
        self.view_menu = QtWidgets.QMenu(self)
        self.view_menu.aboutToShow.connect(self.populate_view_menu)
        self.view_button.setMenu(self.view_menu)
        bar.addWidget(self.view_button)
        help_button = QtWidgets.QPushButton('?')
        help_button.setAccessibleName('Shortcuts and help')
        help_button.setFixedWidth(36)
        help_button.clicked.connect(self.show_help)
        bar.addWidget(help_button)
        return header

    def _build_sidebar(self):
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName('sidebar')
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(560)
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)
        row = QtWidgets.QFrame()
        row.setObjectName('tabRow')
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        self.tab_bar = QtWidgets.QTabBar()
        self.tab_bar.setObjectName('sideTabs')
        self.tab_bar.setExpanding(True)
        self.tab_bar.setDrawBase(False)
        self.tab_bar.currentChanged.connect(self._tab_changed)
        row_layout.addWidget(self.tab_bar, 1)
        collapse = QtWidgets.QPushButton('‹')
        collapse.setObjectName('ghost')
        collapse.setToolTip('Hide the sidebar')
        collapse.setAccessibleName('Hide sidebar')
        collapse.setFixedWidth(36)
        collapse.clicked.connect(self.hide_sidebar)
        row_layout.addWidget(collapse)
        side.addWidget(row)
        self.stack = QtWidgets.QStackedWidget()
        side.addWidget(self.stack, 1)
        return sidebar

    def _build_interval_popup(self):
        popup = QtWidgets.QFrame(self, Qt.Popup)
        popup.setObjectName('popup')
        layout = QtWidgets.QVBoxLayout(popup)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        form = QtWidgets.QFormLayout()
        self.start_edit = self._date_edit()
        self.end_edit = self._date_edit()
        labeled(form, 'Start UTC', self.start_edit)
        labeled(form, 'End UTC', self.end_edit)
        layout.addLayout(form)
        presets = QtWidgets.QHBoxLayout()
        for text, days in (('Now + 1 day', 1), ('Now + 1 lunar day', search.LUNAR_DAY_DAYS), ('Now + 1 year', 365)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _checked=False, d=days: self.set_preset(d))
            presets.addWidget(button)
        layout.addLayout(presets)
        return popup

    def _date_edit(self):
        edit = QtWidgets.QDateTimeEdit()
        edit.setDisplayFormat(EDIT_FORMAT)
        edit.setTimeZone(QtCore.QTimeZone.utc())
        edit.dateTimeChanged.connect(self._popup_edited)
        return edit

    def bind_owner(self, owner):
        self.owner = owner
        saved = owner._saved_interval
        if saved:
            self.interval = (saved[0].replace(microsecond=0), saved[1].replace(microsecond=0))
        else:
            start = sky.utc_now()
            self.interval = (start, start + timedelta(days=search.LUNAR_DAY_DAYS))
        self.search = SearchPanel(self, owner)
        self.tables = TablesPanel(self, owner)
        self.tables.hide()
        self.right.addWidget(self.tables)
        self.right.setStretchFactor(2, 0)
        self.time = TimePanel(self, owner)
        self.points = PointsPanel(self, owner)
        self.add_page('search', 'Search', self.search.widget)
        self.add_page('points', 'Points', self.points)
        self.add_page('time', 'Analysis', self.time)
        owner._visibility_dialog = self.search
        self.set_interval(*self.interval)
        self.refresh_points()
        self.sync_state()

    def refresh_points(self):
        if self.points is None:
            return
        self.points.refresh()
        self.time.sync()
        self.tab_bar.setTabText(self.tab_keys.index('points'), f'Points · {len(self.owner._points)}')

    def eventFilter(self, obj, event):
        if obj is getattr(self, "header", None) and event.type() == QtCore.QEvent.Resize:
            self.position_coordinate_search()
        if (obj is getattr(self, "vtk_widget", None) and event.type() == QtCore.QEvent.Resize
                and self.card is not None and self.card.isVisible()):
            self.card.reposition()
        return super().eventFilter(obj, event)

    def position_coordinate_search(self):
        field = self.coord_field
        field.move(max(0, (self.header.width() - field.width()) // 2),
                   max(0, (self.header.height() - field.height()) // 2))
        field.raise_()

    def set_preview_cursor(self, edge):
        if edge:
            self.vtk_widget.setCursor(Qt.SizeFDiagCursor if edge in ('lt', 'rb') else
                                      Qt.SizeBDiagCursor if edge in ('lb', 'rt') else
                                      Qt.SizeHorCursor if edge in ('l', 'r') else Qt.SizeVerCursor)
        else:
            self.vtk_widget.unsetCursor()

    def add_page(self, key, title, widget):
        self.pages[key] = widget
        self.tab_keys.append(key)
        self.stack.addWidget(widget)
        self.tab_bar.addTab(title)

    def _tab_changed(self, index):
        if 0 <= index < len(self.tab_keys):
            previous = self.current_panel
            self.stack.setCurrentIndex(index)
            self.current_panel = self.tab_keys[index]
            if self.owner is not None:
                if (self.current_panel == 'search' and previous != 'search' and self.search is not None
                        and not self.search._area_search_running):
                    self.search._area_preview_enabled = True
                if self.current_panel != 'search' and self.owner._area_drawing:
                    self.cancel_area_tool()
                if self.current_panel in ('search', 'points') and self.owner._graph_expanded:
                    self.owner.close_graph()
                elif self.current_panel == 'time' and not self.owner._graph_expanded and self.has_analysis():
                    self.owner.reopen_analysis()
                QtCore.QTimer.singleShot(0, self.sync_search_area)

    def sync_search_area(self):
        if self.owner is None or self.search is None:
            return
        visible = (self.current_panel == 'search' and not self.sidebar.isHidden() and not self._zen
                   and not self.owner._graph_expanded and self.search.show_area.isChecked())
        if self.owner._area_drawing and visible:
            return
        bounds = self.search.area_bounds() if visible else None
        if bounds is None:
            self.owner.hide_search_area()
        else:
            self.owner.draw_search_area(bounds)

    def toggle_area_tool(self, checked=False):
        if self.owner is None or self.search is None:
            return
        if self.owner._area_drawing:
            self.cancel_area_tool()
            return
        if self.search._area_search_running:
            self.search.draw_area_button.setChecked(False)
            self.search.status.configure(text='Wait for the current search to finish before drawing a new area.')
            return
        if self.owner._terrain_view:
            self.owner.toggle_terrain_view(render=False)
        self.owner._previous_map_tool = self.owner._map_tool
        self.owner._map_tool = 'Area'; self.owner._area_drawing = True
        self.search._area_preview_enabled = True
        self.search.draw_area_button.setChecked(True)
        self.vtk_widget.setFocus()
        self.search.status.configure(text='Drag on the globe to set search bounds. Right-click cancels.')

    def finish_area_tool(self):
        if self.owner is None or self.search is None:
            return
        self.owner._area_drawing = False
        self.owner._map_tool = self.owner._previous_map_tool
        self.search.draw_area_button.setChecked(False)
        self.search._area_preview_enabled = True
        QtCore.QTimer.singleShot(0, self.sync_search_area)

    def cancel_area_tool(self):
        if self.owner is None or self.search is None:
            return
        self.finish_area_tool()
        self.search.status.configure(text='Area drawing cancelled; search bounds were not changed.')

    def set_interval(self, start, end):
        if self._syncing:
            return
        try:
            start = sky.parse_utc(start).replace(microsecond=0)
            end = sky.parse_utc(end).replace(microsecond=0)
        except ValueError:
            return
        self.interval = (start, end)
        self._syncing = True
        try:
            if self.search is not None:
                self.search.f['start'].setText(start.strftime(TIME_FORMAT))
                self.search.f['end'].setText(end.strftime(TIME_FORMAT))
            for edit, value in ((self.start_edit, start), (self.end_edit, end)):
                edit.setDateTime(QtCore.QDateTime(
                    QtCore.QDate(value.year, value.month, value.day),
                    QtCore.QTime(value.hour, value.minute, value.second), QtCore.QTimeZone.utc()))
            self.interval_button.setText(f'{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC')
        finally:
            self._syncing = False
        if self.time is not None:
            self.time.sync()

    def set_preset(self, days):
        start = sky.utc_now()
        self.set_interval(start, start + timedelta(days=days))

    def _popup_edited(self):
        self.set_interval(self.start_edit.dateTime().toString(EDIT_FORMAT), self.end_edit.dateTime().toString(EDIT_FORMAT))

    def show_interval_popup(self):
        button = self.interval_button
        self.interval_popup.move(button.mapToGlobal(QtCore.QPoint(0, button.height() + 4)))
        self.interval_popup.show()

    def focus_coordinates(self):
        if self.owner is not None and self.owner._graph_expanded:
            return
        if self._zen:
            self.toggle_zen()
        self.coord_field.setFocus()
        self.coord_field.selectAll()

    def go_to_coordinates(self):
        self.go_to(self.coord_field.text())

    def go_to(self, text):
        if self.owner is None:
            return
        try:
            self.owner.lookup_coordinates(text)
        except ValueError as error:
            self.set_coordinate_error(str(error))
            return
        self.set_coordinate_error('')
        self.coord_field.clear()
        self._shown_location = None
        self.vtk_widget.setFocus()
        self.sync_state()

    def set_coordinate_error(self, message):
        field = self.coord_field
        field.setToolTip(message)
        field.setProperty('error', bool(message))
        field.style().unpolish(field)
        field.style().polish(field)
        if message:
            QtWidgets.QToolTip.showText(field.mapToGlobal(QtCore.QPoint(0, field.height())), message, field)

    def sync_state(self):
        owner = self.owner
        if owner is None or self.points is None:
            return
        location = (owner.lat, owner.lon)
        if location != self._shown_location and not self.coord_field.hasFocus():
            self._shown_location = location
            self.set_coordinate_error('')
            self.points.mark_selected()
        self.card.update_state()
        self.search.sync_graph(owner._graph_expanded)
        self.tables.sync()
        self.topbar.sync()
        self.analysis_header.sync()
        self.position_analysis_header()
        for header in self.graph_headers.values():
            header.sync()
        self.timebar.sync()
        self.time.sync()

    def position_analysis_header(self):
        if self.owner is None:
            return
        width = max(1, self.vtk_widget.width())
        height = self.analysis_header.sizeHint().height()
        self.analysis_header.setGeometry(round(.025 * width), round((.04 - self.owner._scroll) * self.vtk_widget.height()), max(1, round(.945 * width)), height)
        self.analysis_header.raise_()

    def has_analysis(self):
        owner = self.owner
        return owner._analysis_inputs is not None and (owner._graph_expanded or owner._analysis_snapshot is not None)

    def analysis_summary(self):
        inputs = self.owner._analysis_inputs
        points = len(inputs['points'])
        where = counted(points, 'point') if points else 'Selected location'
        return f"{where} · {counted(inputs['samples'], 'sample')} · {inputs['method']}"

    def edit_eye_height(self):
        if self.owner is None:
            return
        worker = getattr(self.search, 'worker', None)
        if (worker is not None and worker.is_alive()) or getattr(self.search, '_fill_running', False):
            QtWidgets.QMessageBox.information(self, 'Observer height', 'Wait for the current search to finish before changing the observer height.')
            return
        old = terrain.EYE_HEIGHT_M
        value, accepted = QtWidgets.QInputDialog.getDouble(self, 'Observer height', 'Global height above the terrain (m):', float(old), float(terrain.MIN_EYE_HEIGHT_M), float(terrain.MAX_EYE_HEIGHT_M), 2)
        if not accepted or value == old:
            return
        self.owner.set_eye_height(value)
        if self.search.results:
            self.search.status.configure(text='Observer height changed. Existing search results use the previous height; run the search again.')
        self.owner.window.Render()

    def select_map(self, mode):
        owner = self.owner
        if owner._graph_expanded:
            return
        owner._map_wanted = owner._horizon_wanted = mode
        owner._apply_loaded_maps()
        owner._schedule_map_generation()
        owner.window.Render()

    def set_overlay(self, key, value):
        self.owner._overlay_settings[key] = value
        if key == 'map_legend' and value:
            self.owner._map_legend_open = True
        self.owner.window.Render()

    def set_grid(self, value):
        self.owner._grid_enabled = value
        self.owner.window.Render()

    def reset_view(self):
        if self.owner._terrain_view:
            self.owner.toggle_terrain_view(render=False)
        self.owner.reset_camera()
        self.owner.window.Render()

    def populate_view_menu(self):
        menu = self.view_menu
        menu.clear()
        owner = self.owner

        def command(label, callback, enabled=True):
            action = menu.addAction(label)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _checked=False: callback())

        def option(label, value, callback):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(bool(value))
            action.toggled.connect(callback)

        option('Coordinate grid\tL', owner._grid_enabled, self.set_grid)
        for key, label in (('horizon_inset', 'Horizon graph'), ('horizon_outline', 'Horizon outline'),
                           ('map_legend', 'Map legend'), ('grid_caption', 'Grid information')):
            option(label, owner._overlay_settings[key], lambda value, k=key: self.set_overlay(k, value))
        menu.addSeparator()
        command(f'Observer height… ({terrain.EYE_HEIGHT_M:g} m)\tE', self.edit_eye_height)
        command('Terrain view\tV', owner.toggle_terrain_view)
        command('Reset view\tR', self.reset_view)
        command('Fullscreen\tF', owner.toggle_fullscreen)
        command('Zen mode', self.toggle_zen, not owner._graph_expanded)

    def show_view_menu(self):
        if self.view_button.isVisible():
            self.view_button.showMenu()
        else:
            self.populate_view_menu()
            self.view_menu.popup(self.vtk_widget.mapToGlobal(QtCore.QPoint(16, 16)))

    def build_help(self):
        rows = []
        for title, items in HELP_SECTIONS:
            rows.append(f'<tr><td colspan="2" style="padding-top:12px"><b>{title}</b></td></tr>')
            for keys, text in items:
                rows.append(f'<tr><td style="padding-right:18px"><span style="font-family:\'{MONO_FAMILY}\'">'
                            f'{keys}</span></td><td>{text}</td></tr>')
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Controls')
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(20, 12, 20, 16)
        label = QtWidgets.QLabel(f'<table cellspacing="3">{"".join(rows)}</table>')
        layout.addWidget(label)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(dialog.close)
        layout.addWidget(close, 0, Qt.AlignRight)
        return dialog

    def show_help(self):
        if self.help_dialog is None:
            self.help_dialog = self.build_help()
        self.help_dialog.show()
        self.help_dialog.raise_()

    def show_sidebar(self):
        if self.sidebar.isVisible():
            return
        if self.current_panel == 'search' and self.search is not None and not self.search._area_search_running:
            self.search._area_preview_enabled = True
        self.sidebar.show()
        self.splitter.setSizes([self._sidebar_width, max(1, self.splitter.width() - self._sidebar_width)])
        QtCore.QTimer.singleShot(0, self.sync_search_area)

    def hide_sidebar(self):
        if self.sidebar.isVisible():
            if self.owner is not None and self.owner._area_drawing:
                self.cancel_area_tool()
            self._sidebar_width = self.sidebar.width()
            self.sidebar.hide()
            self.sync_search_area()

    def toggle_sidebar(self):
        if self.sidebar.isVisible():
            self.hide_sidebar()
        else:
            self.show_sidebar()

    def show_panel(self, name):
        if name not in self.pages:
            return
        if self._zen:
            self.toggle_zen()
        index = self.tab_keys.index(name)
        if self.tab_bar.currentIndex() == index:
            self._tab_changed(index)
        else:
            self.tab_bar.setCurrentIndex(index)
        self.show_sidebar()

    def toggle_zen(self):
        if not self._zen and self.owner._graph_expanded:
            return
        self._zen = not self._zen
        self.header.setVisible(not self._zen)
        self.topbar.setVisible(not self._zen)
        self.search.set_zen(self._zen)
        self.tables.sync()
        self.zen_hint.setVisible(self._zen)
        if self._zen:
            self._zen_sidebar = self.sidebar.isVisible()
            self.hide_sidebar()
        elif self._zen_sidebar:
            self.show_sidebar()
        self.owner.set_zen(self._zen)
        self.vtk_widget.setFocus()

    def exit_zen(self):
        if self._zen:
            self.toggle_zen()

    def closeEvent(self, event):
        if self.owner is not None:
            self.owner._sky_stop.set()
            self.search.close()
        event.accept()


def create_application():
    global MONO_FAMILY
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    MONO_FAMILY = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont).family()
    app.setStyleSheet(STYLE.replace('MONO', MONO_FAMILY))
    return app
