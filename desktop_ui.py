"""Integrated Qt shell for the VTK globe, search and interval controls."""
import os
import queue
import sys
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

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
import sky
import terrain
import visibility_search as search

Qt = QtCore.Qt

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
EDIT_FORMAT = 'yyyy-MM-dd HH:mm:ss'
MONO_FAMILY = 'monospace'

STYLE = '''
QWidget { background: transparent; color: #e6edf5; font-size: 13px; }
QMainWindow, QDialog { background: #131b28; }
QFrame#header { background: #0f1622; border-bottom: 1px solid #26334a; }
QFrame#sidebar { background: #1a2433; border-right: 1px solid #26334a; }
QFrame#tabRow { border-bottom: 1px solid #26334a; }
QFrame#popup { background: #1a2433; border: 1px solid #34445a; border-radius: 6px; }
#mono, QDateTimeEdit, QTreeWidget { font-family: "MONO"; }
QLabel#brand { font-size: 14px; font-weight: 600; }
QLabel#groupTitle { font-weight: 600; }
QLabel#muted { color: #9fb0c3; font-size: 12px; }
QLabel#error { color: #ff8f85; }
QPushButton, QToolButton { background: #222e40; border: 1px solid #34445a; border-radius: 6px; padding: 0 12px; min-height: 30px; color: #e6edf5; }
QPushButton:hover, QToolButton:hover { background: #2b3a50; }
QPushButton:pressed, QPushButton:checked, QToolButton:checked { background: #2e3c50; }
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
QTabBar::tab { background: transparent; color: #9fb0c3; padding: 0 14px; min-height: 32px; border: 0; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #e6edf5; border-bottom-color: #2bb5c5; }
QTabBar#sideTabs::tab { min-height: 40px; font-weight: 500; }
QScrollArea { border: 0; }
QFrame#pointRow { border-radius: 8px; }
QFrame#pointRow[on="true"] { background: #222e40; }
QFrame#card { background: #1a2433; border: 1px solid #2e3c50; border-radius: 8px; }
QFrame#barTop { border-top: 1px solid #26334a; }
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


class MirrorLabel:
    def __init__(self, *labels): self.labels = labels
    def configure(self, **kw):
        for label in self.labels: label.configure(**kw)


class Button(QtWidgets.QPushButton):
    def configure(self, **kw):
        if 'text' in kw: self.setText(str(kw['text']))
        if 'state' in kw: self.setEnabled(kw['state'] != 'disabled')


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
        if ids: self.clear()
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
        self.addTopLevelItem(item)
    def selection(self):
        chosen = {str(item.data(0, Qt.UserRole)) for item in self.selectedItems()}
        return tuple(i for i in self.get_children() if i in chosen)
    def selection_set(self, ids):
        if isinstance(ids, str): ids = (ids,)
        wanted = set(map(str, ids))
        self.clearSelection()
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if str(item.data(0, Qt.UserRole)) in wanted: item.setSelected(True)
    def see(self, iid):
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if str(item.data(0, Qt.UserRole)) == str(iid): self.scrollToItem(item); break


class TabAdapter:
    def __init__(self, tabs): self.tabs = tabs
    def select(self, page): self.tabs.setCurrentWidget(page)


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
    def deiconify(self): self.panel.shell.show_panel('search')
    def lift(self): pass
    def focus_force(self): pass
    def destroy(self): self.panel.widget.hide()


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
        self.tk = SimpleNamespace(TclError=RuntimeError)
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
        self._display_items, self.last_parameters, self._refinement_sectors = [], None, None
        self._sort = self._cluster_cache = None
        self.finished = self.was_cancelled = False
        outer = QtWidgets.QVBoxLayout(self.widget)
        outer.setContentsMargins(16, 8, 16, 12)
        outer.setSpacing(8)
        self.page_tabs = QtWidgets.QTabWidget()
        self.search_page = QtWidgets.QWidget(); self.results_page = QtWidgets.QWidget()
        self.page_tabs.addTab(self.search_page, 'Criteria')
        self.page_tabs.addTab(self.results_page, 'Results')
        outer.addWidget(self.page_tabs, 1)
        self.workspace_tabs = TabAdapter(self.page_tabs)
        self.workspace_settings, self.workspace_results = self.search_page, self.results_page
        self.settings_tabs, self.area_tab = self.workspace_tabs, self.search_page
        self._build_criteria()
        self._build_results()
        self.status = MirrorLabel(self.criteria_status, self.results_status)
        self._update_lunar_state()
        self._refresh_headings()
        if owner._search_session is not None: self.restore(owner._search_session)
        self.poll()

    def _build_criteria(self):
        container = QtWidgets.QVBoxLayout(self.search_page)
        container.setContentsMargins(0, 8, 0, 0)
        self.use_terrain = Check('Use terrain horizon')
        self.use_terrain.set(True)
        container.addWidget(self.use_terrain)
        mode_note = QtWidgets.QLabel('Off: find optimistic candidate clusters first.')
        mode_note.setObjectName('muted'); mode_note.setWordWrap(True); container.addWidget(mode_note)
        scroll = self.criteria_scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget(); stack = QtWidgets.QVBoxLayout(body)
        stack.setContentsMargins(3, 3, 6, 6); stack.setSpacing(8)
        self.f = {}
        def field(form, key, label, value):
            entry = Entry(str(value)); self.f[key] = entry; labeled(form, label, entry)
        for name, value in zip(('start', 'end'), self.shell.interval):
            entry = Entry(value.strftime(TIME_FORMAT), self.search_page)
            entry.hide()
            self.f[name] = entry
        f = QtWidgets.QFormLayout()
        field(f, 'step', 'Step (hours)', '2')
        stack.addWidget(group('Time', f))
        f = QtWidgets.QFormLayout()
        for key, label, value in [('min_earth','Earth minimum %','50'),('max_earth','Earth maximum %','100'),
                                  ('min_sun','Sun minimum %','30'),('max_sun','Sun maximum %','100')]: field(f,key,label,value)
        note = QtWidgets.QLabel('Maximums apply to terrain searches.'); note.setObjectName('muted'); note.setWordWrap(True); f.addRow(note)
        stack.addWidget(group('Visibility', f))
        f = QtWidgets.QFormLayout()
        for key,label,value in [('spacing','Grid spacing (deg)','10'),('lat','Latitude min, max','-90, 90'),
                                 ('lon','Longitude start, end','-180, 180')]: field(f,key,label,value)
        self.area_box = group('Area', f); stack.addWidget(self.area_box)
        f = QtWidgets.QFormLayout()
        field(f,'max_slope','Maximum slope (deg; optional)','')
        field(f,'slope_radius','Slope radius (km)','1')
        stack.addWidget(group('Slope', f))
        f = QtWidgets.QFormLayout()
        field(f,'side','Samples per side','3')
        field(f,'horizon','Pessimistic horizon (deg)','2')
        self.screen_mode = Combo(); self.screen_mode.addItems(['Pessimistic center','All cells'])
        self.range_mode = Combo(); self.range_mode.addItems(['On','Off'])
        labeled(f,'Cell screening',self.screen_mode); labeled(f,'Optimistic range',self.range_mode)
        note = QtWidgets.QLabel('Screening is approximate and can miss suitable cells.'); note.setObjectName('muted'); note.setWordWrap(True); f.addRow(note)
        stack.addWidget(group('Screening', f))
        f = QtWidgets.QFormLayout()
        self.lunar_avg_check = self.lunar_avg = Check('Calculate and show average Sun % per lunar day')
        self.lunar_sd_check = self.lunar_sd = Check('Calculate and show standard deviation')
        self.lunar_sd_check.setToolTip('Enable the average first. Available for terrain searches longer than two days.')
        f.addRow(self.lunar_avg_check); f.addRow(self.lunar_sd_check)
        for key,label in [('lunar_min_avg','Minimum average %'),('lunar_max_avg','Maximum average %'),
                          ('lunar_min_sd','Minimum SD %'),('lunar_max_sd','Maximum SD %')]: field(f,key,label,'')
        self.lunar_note = Label(); self.lunar_note.setObjectName('muted'); self.lunar_note.setWordWrap(True); f.addRow(self.lunar_note)
        stack.addWidget(group('Lunar-day statistics',f)); stack.addStretch(1)
        scroll.setWidget(body); container.addWidget(scroll,1)
        for name in ('start','end'): self.f[name].textChanged.connect(self._interval_edited)
        self.lunar_avg_check.clicked.connect(self._update_lunar_state)
        self.lunar_sd_check.clicked.connect(self._update_lunar_state)
        self.use_terrain.clicked.connect(self._update_lunar_state)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = Button('Start search'); self.start_button.setObjectName('primary'); self.start_button.clicked.connect(self.start)
        self.cancel_button = Button('Cancel'); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel.set)
        buttons.addWidget(self.start_button); buttons.addWidget(self.cancel_button); container.addLayout(buttons)
        self.progress = Progress(); self.progress.setTextVisible(False); container.addWidget(self.progress)
        self.criteria_status = Label('Choose criteria and start the search.'); self.criteria_status.setWordWrap(True); container.addWidget(self.criteria_status)
        self.details = QtWidgets.QWidget(); detail_layout = QtWidgets.QVBoxLayout(self.details); detail_layout.setContentsMargins(0,0,0,0)
        self.fill_label = Label(); self.cache_label = Label(); self.range_label = Label(); self.horizon_label = Label()
        for label in (self.fill_label,self.cache_label,self.range_label,self.horizon_label):
            label.setObjectName('muted'); label.setWordWrap(True); detail_layout.addWidget(label)
        self.details.hide()
        details_button=QtWidgets.QPushButton('Calculation details'); details_button.setCheckable(True)
        details_button.toggled.connect(self.details.setVisible); container.addWidget(details_button); container.addWidget(self.details)

    def _interval_edited(self):
        self.shell.set_interval(self.f['start'].get(), self.f['end'].get())
        self._update_lunar_state()

    def _build_results(self):
        root=QtWidgets.QVBoxLayout(self.results_page); root.setContentsMargins(0,8,0,0)
        bar=QtWidgets.QHBoxLayout(); bar.addWidget(QtWidgets.QLabel('View'))
        self.view_mode=Combo(); self.view_mode.addItems(['Points','Clusters']); self.view_mode.currentTextChanged.connect(lambda _: self.populate_results('Showing') if self.results else None)
        bar.addWidget(self.view_mode,1)
        columns=QtWidgets.QPushButton('Columns'); columns.setCheckable(True); bar.addWidget(columns); root.addLayout(bar)
        self._columns=tuple(spec[0] for spec in search.COLUMN_SPECS)
        self._heading_labels={spec[0]:spec[1] for spec in search.COLUMN_SPECS}
        self.column_vars={}
        column_panel=QtWidgets.QWidget(); colgrid=QtWidgets.QGridLayout(column_panel); colgrid.setContentsMargins(2,2,2,2)
        visible_keys=search.PERCENT_COLUMNS
        default_keys=search.DEFAULT_COLUMNS
        visible_index=0
        for spec in search.COLUMN_SPECS:
            checkbox=Check(spec[1]); checkbox.set(spec[0] in default_keys); checkbox.clicked.connect(self._columns_changed)
            self.column_vars[spec[0]]=checkbox
            if spec[0] in visible_keys:
                colgrid.addWidget(checkbox,visible_index//2,visible_index%2);visible_index+=1
        colscroll=QtWidgets.QScrollArea();colscroll.setWidgetResizable(True);colscroll.setWidget(column_panel);colscroll.setMaximumHeight(190);colscroll.hide()
        columns.toggled.connect(colscroll.setVisible); root.addWidget(colscroll)
        self.table=Tree(self._columns)
        for spec in search.COLUMN_SPECS:self.table.column(spec[0],width=spec[3])
        self.table.itemDoubleClicked.connect(lambda *_: self.view_selected())
        root.addWidget(self.table,1)
        self.results_status=Label('Results will appear here.');self.results_status.setWordWrap(True);root.addWidget(self.results_status)
        hint=QtWidgets.QLabel('Shift / Ctrl to select multiple points. Select a cluster to refine it.')
        hint.setObjectName('muted');hint.setWordWrap(True);root.addWidget(hint)
        grid=QtWidgets.QGridLayout()
        self.view_button=Button('View selected');self.view_button.clicked.connect(self.view_selected)
        self.select_all_button=Button('Select all');self.select_all_button.clicked.connect(self.select_all)
        self.add_button=Button('Add points');self.add_button.clicked.connect(self.add_selected)
        self.analyze_button=Button('Compare');self.analyze_button.clicked.connect(self.analyze_selected)
        self.marks_button=Button('Show on globe');self.marks_button.clicked.connect(self.show_all)
        self.sector_button=Button('Search cluster');self.sector_button.clicked.connect(self.refine_selected_sector)
        self.export_button=Button('Save CSV');self.export_button.clicked.connect(self.export)
        self.clear_button=Button('Clear markers');self.clear_button.clicked.connect(self.clear)
        buttons=(self.view_button,self.select_all_button,self.add_button,self.analyze_button,
                 self.marks_button,self.sector_button,self.export_button,self.clear_button)
        for i,button in enumerate(buttons):
            grid.addWidget(button,i//2,i%2)
            if button is not self.clear_button:button.setEnabled(False)
        root.addLayout(grid)

    def export(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self.widget,'Save results','','CSV files (*.csv)')
        if path:
            if not path.lower().endswith('.csv'): path += '.csv'
            self.export_to(path)

    def refine_selected_sector(self):
        super().refine_selected_sector()
        if self._refinement_sectors is not None:
            self.criteria_scroll.ensureWidgetVisible(self.area_box)

    def close(self):
        if self._closed: return
        self._closed=True; self.cancel.set(); self._fill_cancel.set()
        self.window.after_cancel(self.timer)
        self.widget.hide()


class TimePanel(QtWidgets.QWidget):
    def __init__(self, shell, owner):
        super().__init__()
        self.shell, self.owner = shell, owner
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        form = QtWidgets.QFormLayout()
        self.step = Entry()
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(['seconds', 'minutes', 'hours', 'days'])
        self.method = QtWidgets.QComboBox()
        self.method.addItems(list(sky.SKY_METHODS))
        labeled(form, 'Step', self.step)
        labeled(form, 'Step unit', self.unit)
        labeled(form, 'Model', self.method)
        root.addWidget(group('Analysis', form))
        note = QtWidgets.QLabel('Analyzes the selected location or comparison points over the interval set in the header.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        root.addWidget(note)
        self.error = QtWidgets.QLabel()
        self.error.setObjectName('error')
        self.error.setWordWrap(True)
        root.addWidget(self.error)
        apply = QtWidgets.QPushButton('Analyze interval')
        apply.setObjectName('primary')
        apply.clicked.connect(self.apply)
        root.addWidget(apply)
        self.reopen_button = QtWidgets.QPushButton('Reopen last analysis')
        self.reopen_button.setEnabled(False)
        self.reopen_button.clicked.connect(owner.reopen_analysis)
        root.addWidget(self.reopen_button)
        root.addStretch(1)
        self.load_saved()

    def load_saved(self):
        saved = self.owner._saved_interval
        if saved:
            _, _, step, unit, method = saved
        else:
            step, unit, method = 60, 'minutes', self.owner._sky_method
        self.step.setText(str(step))
        self.unit.setCurrentText(unit)
        self.method.setCurrentText(method)

    def apply(self):
        start, stop = (value.strftime(TIME_FORMAT) for value in self.shell.interval)
        try:
            self.owner.set_time_interval(start, stop, float(self.step.text()), self.unit.currentText(), self.method.currentText())
        except (ValueError, OverflowError) as error:
            self.error.setText(str(error))
        else:
            self.error.clear()


class CoordinateField(QtWidgets.QLineEdit):
    escaped = QtCore.Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.escaped.emit()
            return
        super().keyPressEvent(event)


HELP_SECTIONS = (
    ('Mouse', (('Drag', 'Orbit'), ('Middle / Shift-drag', 'Pan'), ('Ctrl-drag', 'Orbit around the cursor point'),
               ('Wheel', 'Zoom'), ('Click', 'Select a location'), ('Shift-click', 'Add a comparison point'))),
    ('Navigation', (('G', 'Go to coordinates'), ('V', 'Terrain view'), ('R', 'Reset view'), ('F', 'Fullscreen'),
                    ('Esc', 'Exit zen mode'))),
    ('Search and analysis', (('S', 'Search tab'), ('T', 'Analysis tab'), ('I', 'Analyze selected result'),
                             ('A', 'Add point'), ('P', 'Manage points'),
                             (', / .', 'Previous / next comparison point'), ('[ / ]', 'Previous / next time step'))),
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
        self.check = QtWidgets.QCheckBox()
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
        self.name.setStyleSheet(f'font-weight: 500; color: {"#e6edf5" if included else "#9fb0c3"};')
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
        row.addWidget(icon_button('target', f'Go to point {index + 1}', lambda: panel.go_to(point)))
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
        self.empty = QtWidgets.QLabel('Click the Moon and press Add to points, or add results from a search. Shift-click adds directly.')
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
        menu.addAction('Add to current points').triggered.connect(lambda: self.load(True))
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
        self.analyze_button.setText(f'Analyze {ticked} point{"" if ticked == 1 else "s"}')
        self.analyze_button.setEnabled(bool(ticked))
        self.analyze_button.setToolTip('' if ticked else 'Tick at least one point to analyze.')
        self.mark_selected()

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
        if owner.point_index(owner.lat, owner.lon) is None:
            owner.add_current_point()
        else:
            owner.toggle_terrain_view()

    def update_state(self):
        owner = self.shell.owner
        visible = not (owner._graph_expanded or owner._terrain_view or self.shell._zen)
        if visible:
            index = owner.point_index(owner.lat, owner.lon)
            full = len(owner._points) >= compare.MAX_POINTS
            key = (owner.lat, owner.lon, index, owner._point_names.get(owner._points[index]) if index is not None else None,
                   full, owner._selected_stats)
            if key != self.key:
                self.key = key
                self.refill(index, full)
        if visible != self.isVisible():
            self.setVisible(visible)
        if visible:
            self.reposition()

    def refill(self, index, full):
        owner = self.shell.owner
        elevation, slope, low, high = owner._selected_stats
        if index is None:
            self.title.setText('Selected location')
            self.button.setText('Add to points')
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
        self.tag.setVisible(index is None)
        self.button.style().unpolish(self.button)
        self.button.style().polish(self.button)
        self.coords.setText(f'{owner.lat:.2f}°, {owner.lon:.2f}°')
        self.values['Elevation'].setText(f'{elevation:+,.0f} m'.replace(',', ' '))
        self.values['Slope'].setText(f'{slope:.1f}°')
        self.values['Horizon'].setText(f'{low:+.2f}° … {high:+.2f}°')
        self.adjustSize()

    def reposition(self):
        self.move(self.parentWidget().width() - self.width() - 16, 16)
        self.raise_()


class MainShell(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Moon Horizon')
        self.resize(1440, 900)
        self.owner = None
        self.search = None
        self.time = None
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
        outer.addWidget(self.header)
        self.zen_hint = QtWidgets.QPushButton('Exit zen mode  ·  Esc')
        self.zen_hint.clicked.connect(self.toggle_zen)
        self.zen_hint.hide()
        outer.addWidget(self.zen_hint, 0, Qt.AlignRight)
        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(3)
        self.sidebar = self._build_sidebar()
        self.vtk_widget = GlobeWidget(self.splitter)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.vtk_widget)
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
        brand = QtWidgets.QLabel('Moon Horizon')
        brand.setObjectName('brand')
        bar.addWidget(brand)
        self.coord_field = CoordinateField()
        self.coord_field.setObjectName('mono')
        self.coord_field.setPlaceholderText('lat, lon')
        self.coord_field.setAccessibleName('Go to latitude, longitude')
        self.coord_field.setFixedWidth(190)
        self.coord_field.returnPressed.connect(self.go_to_coordinates)
        self.coord_field.textEdited.connect(lambda _text: self.set_coordinate_error(''))
        self.coord_field.escaped.connect(lambda: self.vtk_widget.setFocus())
        bar.addWidget(self.coord_field)
        self.interval_button = QtWidgets.QPushButton()
        self.interval_button.setObjectName('mono')
        self.interval_button.setToolTip('Time interval used by search and analysis')
        self.interval_button.clicked.connect(self.show_interval_popup)
        bar.addWidget(self.interval_button)
        bar.addStretch(1)
        self.view_button = QtWidgets.QPushButton('View')
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
        self.tab_bar.setTabText(self.tab_keys.index('points'), f'Points · {len(self.owner._points)}')

    def eventFilter(self, obj, event):
        if obj is self.vtk_widget and event.type() == QtCore.QEvent.Resize and self.card.isVisible():
            self.card.reposition()
        return super().eventFilter(obj, event)

    def add_page(self, key, title, widget):
        self.pages[key] = widget
        self.tab_keys.append(key)
        self.stack.addWidget(widget)
        self.tab_bar.addTab(title)

    def _tab_changed(self, index):
        if 0 <= index < len(self.tab_keys):
            self.stack.setCurrentIndex(index)
            self.current_panel = self.tab_keys[index]

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
        self.time.reopen_button.setEnabled(owner._analysis_can_reopen and not owner._graph_expanded)
        location = (owner.lat, owner.lon)
        if location != self._shown_location and not self.coord_field.hasFocus():
            self._shown_location = location
            self.coord_field.setText(f'{owner.lat:.4f}, {owner.lon:.4f}')
            self.set_coordinate_error('')
            self.points.mark_selected()
        self.card.update_state()

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

        modes = QtGui.QActionGroup(menu)
        for mode in ('Color', 'Elevation', 'Slope'):
            action = menu.addAction(mode)
            action.setActionGroup(modes)
            action.setCheckable(True)
            action.setChecked(owner._map_wanted == mode)
            action.triggered.connect(lambda _checked=False, m=mode: self.select_map(m))
        menu.addSeparator()
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
        self.sidebar.show()
        self.splitter.setSizes([self._sidebar_width, max(1, self.splitter.width() - self._sidebar_width)])

    def hide_sidebar(self):
        if self.sidebar.isVisible():
            self._sidebar_width = self.sidebar.width()
            self.sidebar.hide()

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
        self.tab_bar.setCurrentIndex(self.tab_keys.index(name))
        self.show_sidebar()

    def toggle_zen(self):
        if not self._zen and self.owner._graph_expanded:
            return
        self._zen = not self._zen
        self.header.setVisible(not self._zen)
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
