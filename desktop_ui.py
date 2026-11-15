"""Integrated Qt shell for the VTK globe, search and interval controls."""
import os
import queue
import sys
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

# QVTK's X11 render window needs an X11 widget handle. Under a Wayland
# desktop, use XWayland when available unless the user chose a Qt platform.
if (sys.platform.startswith('linux') and os.environ.get('WAYLAND_DISPLAY')
        and os.environ.get('DISPLAY')):
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import vtk

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    QT_BINDING = 'PySide6'
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        QT_BINDING = 'PyQt5'
    except ImportError as error:
        raise ImportError('Install PySide6 or PyQt5 to use the integrated interface.') from error

import vtkmodules.qt
vtkmodules.qt.PyQtImpl = QT_BINDING
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import sky
import terrain
import visibility_search as search

Qt = QtCore.Qt

STYLE = '''
QMainWindow, QWidget { background: #172030; color: #e6edf7; font-size: 13px; }
QFrame#header { background: #111a28; border-bottom: 1px solid #495057; }
QFrame#sidebar { background: #1d2b3d; border-right: 1px solid #525b63; }
QLabel#brand { font-size: 16px; font-weight: 700; letter-spacing: 1px; color: #f3f4f5; }
QLabel#sectionTitle { font-size: 18px; font-weight: 650; color: #ffffff; }
QLabel#muted { color: #b4bec5; }
QPushButton, QToolButton { background: #2b3b51; border: 1px solid #455b76; border-radius: 4px; padding: 7px 10px; color: #f0f3f5; }
QPushButton:hover, QToolButton:hover { background: #394e68; }
QPushButton:pressed, QToolButton:checked { background: #17677a; border-color: #49c5d6; }
QPushButton:disabled { color: #8a959d; background: #2d343a; border-color: #404b52; }
QPushButton#primary { background: #087c8d; border-color: #35c2d4; font-weight: 650; }
QPushButton#primary:hover { background: #2192a2; }
QLineEdit, QComboBox, QTreeWidget { background: #111b2b; color: #f4f6f7; border: 1px solid #486078; border-radius: 3px; padding: 5px; selection-background-color: #17677a; }
QComboBox::drop-down { border: 0; width: 22px; }
QTreeWidget { padding: 0; alternate-background-color: #253448; }
QHeaderView::section { background: #2b3b51; color: #e7edf0; border: 0; border-right: 1px solid #58626a; padding: 6px; }
QGroupBox { border: 1px solid #56616a; border-radius: 4px; margin-top: 12px; padding: 12px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; color: #dbe5e8; }
QTabWidget::pane { border: 1px solid #4d5962; background: #1d2b3d; }
QTabBar::tab { background: #252b30; color: #b8c4ca; padding: 8px 12px; border: 1px solid #4d5962; }
QTabBar::tab:selected { background: #35454e; color: white; border-bottom: 2px solid #56bdca; }
QScrollArea { border: 0; }
QProgressBar { border: 1px solid #57636a; border-radius: 3px; text-align: center; background: #1d2328; min-height: 12px; }
QProgressBar::chunk { background: #45a6b5; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 17px; height: 17px; border: 2px solid #94aac4; border-radius: 3px; background: #101a2b; }
QCheckBox::indicator:checked { background: #35d0df; border: 2px solid #b4f8ff; }
QCheckBox::indicator:hover { border: 2px solid #f5ffff; }
QCheckBox:disabled { color: #718096; }
QCheckBox::indicator:disabled { background: #273341; border: 2px solid #536171; }
QCheckBox::indicator:checked:disabled { background: #4c6870; border: 2px solid #6a8086; }
QSplitter::handle { background: #40536b; width: 3px; }
QMenu { background: #1d2b3d; color: #e6edf7; border: 1px solid #455b76; padding: 5px; }
QMenu::item { padding: 6px 24px 6px 12px; }
QMenu::item:selected { background: #394e68; }
QPushButton#closeButton { border: 1px solid transparent; background: transparent; padding: 4px 8px; }
QPushButton#closeButton:hover { background: #40556f; border-color: #6d829e; }
QPushButton#reopenButton { font-size: 14px; padding: 2px; background: #24374c; border: 1px solid #7990a8; border-radius: 4px; }
QPushButton#reopenButton:hover { background: #3b6078; border-color: #e4f9ff; }
'''
class Entry(QtWidgets.QLineEdit):
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
        color = QtGui.QColor('#062a35' if self.isEnabled() else '#a6b8c0')
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
    box = QtWidgets.QGroupBox(title)
    box.setLayout(layout)
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
        outer.setContentsMargins(14, 14, 14, 12)
        outer.setSpacing(8)
        title = QtWidgets.QLabel('Visibility search')
        title.setObjectName('sectionTitle')
        outer.addWidget(title)
        subtitle = QtWidgets.QLabel('Set criteria, search, then explore matching areas or points.')
        subtitle.setObjectName('muted'); subtitle.setWordWrap(True); outer.addWidget(subtitle)
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
        scroll = self.criteria_scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget(); stack = QtWidgets.QVBoxLayout(body)
        stack.setContentsMargins(3, 3, 6, 6); stack.setSpacing(8)
        self.f = {}
        now = sky.utc_now()
        def field(form, key, label, value):
            entry = Entry(str(value)); self.f[key] = entry; labeled(form, label, entry)
        f = QtWidgets.QFormLayout()
        field(f, 'start', 'Start UTC', now.strftime('%Y-%m-%d %H:%M:%S'))
        field(f, 'end', 'End UTC', (now + timedelta(days=search.LUNAR_DAY_DAYS)).strftime('%Y-%m-%d %H:%M:%S'))
        field(f, 'step', 'Step (hours)', '2')
        stack.addWidget(group('Time interval', f))
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
        for name in ('start','end'): self.f[name].textChanged.connect(self._update_lunar_state)
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
    def __init__(self, owner):
        super().__init__(); self.owner=owner
        root=QtWidgets.QVBoxLayout(self);root.setContentsMargins(14,14,14,14);root.setSpacing(10)
        title=QtWidgets.QLabel('Time interval');title.setObjectName('sectionTitle');root.addWidget(title)
        subtitle=QtWidgets.QLabel('Choose the UTC interval for the selected location or comparison points.');subtitle.setObjectName('muted');subtitle.setWordWrap(True);root.addWidget(subtitle)
        form=QtWidgets.QFormLayout();self.start=QtWidgets.QLineEdit();self.stop=QtWidgets.QLineEdit();self.step=QtWidgets.QLineEdit()
        self.unit=QtWidgets.QComboBox();self.unit.addItems(['seconds','minutes','hours','days'])
        self.method=QtWidgets.QComboBox();self.method.addItems(list(sky.SKY_METHODS))
        labeled(form,'Start UTC',self.start);labeled(form,'End UTC',self.stop);labeled(form,'Step',self.step);labeled(form,'Step unit',self.unit);labeled(form,'Model',self.method)
        root.addWidget(group('Analysis period',form))
        note=QtWidgets.QLabel('Format: YYYY-MM-DD HH:MM:SS (UTC)');note.setObjectName('muted');note.setWordWrap(True);root.addWidget(note)
        self.error=QtWidgets.QLabel();self.error.setWordWrap(True);self.error.setStyleSheet('color: #ffb3a9;');root.addWidget(self.error)
        apply=QtWidgets.QPushButton('Analyze interval');apply.setObjectName('primary');apply.clicked.connect(self.apply);root.addWidget(apply)
        root.addStretch(1)
        self.load_saved()
    def load_saved(self):
        saved=self.owner._saved_interval
        if saved: start,stop,step,unit,method=saved
        else: start=sky.utc_now();stop=start+timedelta(days=1);step=60;unit='minutes';method=self.owner._sky_method
        self.start.setText(start.strftime('%Y-%m-%d %H:%M:%S'));self.stop.setText(stop.strftime('%Y-%m-%d %H:%M:%S'))
        self.step.setText(str(step));self.unit.setCurrentText(unit);self.method.setCurrentText(method)
    def apply(self):
        try:self.owner.set_time_interval(self.start.text(),self.stop.text(),float(self.step.text()),self.unit.currentText(),self.method.currentText())
        except (ValueError,OverflowError) as error:self.error.setText(str(error))
        else:self.error.clear()


class LayersPanel(QtWidgets.QWidget):
    def __init__(self, owner):
        super().__init__();root=QtWidgets.QVBoxLayout(self);root.setContentsMargins(14,14,14,14)
        title=QtWidgets.QLabel('Display');title.setObjectName('sectionTitle');root.addWidget(title)
        intro=QtWidgets.QLabel('Choose the information shown over the globe.');intro.setObjectName('muted');intro.setWordWrap(True);root.addWidget(intro)
        layout=QtWidgets.QVBoxLayout()
        labels=(('help','Navigation guide'),('horizon_inset','Horizon graph'),('horizon_outline','Horizon outline'),
                ('point_details','Point details'),('grid_caption','Grid information'),('points_caption','Comparison points information'),
                ('map_status','Map information'),('map_legend','Map legend'))
        for key,label in labels:
            box=Check(label);box.setChecked(owner._overlay_settings[key])
            box.toggled.connect(lambda value,k=key:(owner._overlay_settings.__setitem__(k,value),owner.window.Render()))
            layout.addWidget(box)
        root.addWidget(group('Visible layers',layout))
        change=QtWidgets.QPushButton('Change globe map');change.clicked.connect(owner.cycle_map_mode);root.addWidget(change)
        shortcuts=QtWidgets.QLabel('S  Search    T  Time    M  Map    H  Display\nG  Coordinates    V  Terrain    P  Points\nShift-click / A  Add point    R  Reset')
        shortcuts.setObjectName('muted');shortcuts.setWordWrap(True);root.addWidget(shortcuts);root.addStretch(1)


class MainShell(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('Moon Horizon');self.resize(1440,900)
        self.owner=None;self.current_panel=None
        root=QtWidgets.QWidget();self.setCentralWidget(root);outer=QtWidgets.QVBoxLayout(root);outer.setContentsMargins(0,0,0,0);outer.setSpacing(0)
        header=QtWidgets.QFrame();self.header=header;header.setObjectName('header');bar=QtWidgets.QHBoxLayout(header);bar.setContentsMargins(14,8,14,8)
        self.nav_buttons={}
        for label,key in [('Search','search'),('Time analysis','time')]:
            button=QtWidgets.QPushButton(label);button.setCheckable(True)
            button.clicked.connect(lambda _checked=False,k=key:self.toggle_panel(k));bar.addWidget(button);self.nav_buttons[key]=button
        self.observer_button=QtWidgets.QPushButton('Observer height…');self.observer_button.setToolTip('Global height above the terrain');self.observer_button.clicked.connect(self.edit_eye_height);bar.addWidget(self.observer_button)
        self.map_button=QtWidgets.QPushButton('Layers');self.map_menu=QtWidgets.QMenu(self);self.map_button.setMenu(self.map_menu);bar.addWidget(self.map_button)
        self.reopen_buttons={};self._closed_panels=set()
        bar.addStretch(1)
        self.controls_button=QtWidgets.QPushButton('Controls');self.controls_menu=QtWidgets.QMenu(self);self.controls_button.setMenu(self.controls_menu);bar.addWidget(self.controls_button)
        self.zen_button=QtWidgets.QPushButton('◌');self.zen_button.setToolTip('Zen mode');self.zen_button.setAccessibleName('Zen mode');self.zen_button.clicked.connect(self.toggle_zen);bar.addWidget(self.zen_button)
        outer.addWidget(header)
        self.zen_hint=QtWidgets.QPushButton('Exit zen mode  ·  Esc');self.zen_hint.clicked.connect(self.toggle_zen);self.zen_hint.hide();outer.addWidget(self.zen_hint,0,Qt.AlignRight);self._zen=False
        self.splitter=QtWidgets.QSplitter(Qt.Horizontal);self.sidebar=QtWidgets.QFrame();self.sidebar.setObjectName('sidebar');self.sidebar.setMinimumWidth(340);self.sidebar.setMaximumWidth(760)
        self.stack=QtWidgets.QStackedWidget();side=QtWidgets.QVBoxLayout(self.sidebar);side.setContentsMargins(0,0,0,0)
        close_row=QtWidgets.QHBoxLayout();close_row.addStretch(1);close=QtWidgets.QPushButton('×');close.setObjectName('closeButton');close.setToolTip('Close panel');close.setAccessibleName('Close panel');close.clicked.connect(self.close_panel);close_row.addWidget(close);side.addLayout(close_row);side.addWidget(self.stack)
        self.vtk_widget=QVTKRenderWindowInteractor(self.splitter)
        for key,label,side in [('search','Reopen search','left'),('time','Reopen time settings','left'),('horizon','Reopen horizon graph','right'),('legend','Reopen map legend','right'),('analysis','Reopen time analysis','right')]:
            button=QtWidgets.QPushButton('▶' if side=='left' else '◀',self.vtk_widget)
            button.setObjectName('reopenButton');button.setFixedSize(27,30);button.setToolTip(label);button.setAccessibleName(label);button.hide();self.reopen_buttons[key]=button
        self.reopen_buttons['search'].clicked.connect(lambda:self.show_panel('search'))
        self.reopen_buttons['time'].clicked.connect(lambda:self.show_panel('time'))
        self.reopen_buttons['horizon'].clicked.connect(lambda:self.set_overlay('horizon_inset',True))
        self.reopen_buttons['legend'].clicked.connect(lambda:self.set_overlay('map_legend',True))
        self.reopen_buttons['analysis'].clicked.connect(self.reopen_analysis)
        self.vtk_widget.installEventFilter(self)
        self.splitter.addWidget(self.sidebar);self.splitter.addWidget(self.vtk_widget);self.splitter.setStretchFactor(0,0);self.splitter.setStretchFactor(1,1)
        outer.addWidget(self.splitter,1);self.sidebar.hide()
        shortcut_type=getattr(QtGui,'QShortcut',None) or QtWidgets.QShortcut
        self.zen_shortcut=shortcut_type(QtGui.QKeySequence('Escape'),self);self.zen_shortcut.activated.connect(self.exit_zen)
    def bind_owner(self,owner):
        self.owner=owner
        self.search=SearchPanel(self,owner);self.time=TimePanel(owner)
        self.pages={'search':self.search.widget,'time':self.time}
        for page in self.pages.values():self.stack.addWidget(page)
        owner._visibility_dialog=self.search
        self.map_menu.aboutToShow.connect(self.populate_maps);self.controls_menu.aboutToShow.connect(self.populate_controls);self.update_map_button();self.update_observer_button();self.sync_reopen_buttons()
    def update_observer_button(self):
        self.observer_button.setText(f'Observer · {terrain.EYE_HEIGHT_M:g} m')
    def sync_reopen_buttons(self):
        if self.owner is None:return
        owner=self.owner; expanded=owner._graph_expanded
        states={key:not expanded and key in self._closed_panels and self.current_panel != key for key in ('search','time')}
        states['horizon']=not expanded and not owner._overlay_settings['horizon_inset']
        states['legend']=not expanded and owner._map_mode in ('Elevation','Slope') and not owner._overlay_settings['map_legend']
        states['analysis']=not expanded and owner._analysis_can_reopen
        for key,show in states.items():
            show=show and not self._zen
            button=self.reopen_buttons[key];was_hidden=button.isHidden();button.setVisible(show)
            if show and was_hidden:button.raise_()
        self.position_reopen_buttons()
    def position_reopen_buttons(self):
        width=max(1,self.vtk_widget.width());height=max(1,self.vtk_widget.height())
        positions={'search':(0,56),'time':(0,92),'analysis':(width-27,36),'legend':(width-27,int(height*.43)),'horizon':(width-27,int(height*.72))}
        for key,button in self.reopen_buttons.items():
            x,y=positions[key];button.move(max(0,x),max(0,min(height-30,y)))
    def eventFilter(self,obj,event):
        if obj is self.vtk_widget and event.type()==QtCore.QEvent.Resize:self.position_reopen_buttons()
        return super().eventFilter(obj,event)
    def reopen_analysis(self):
        if self.owner._analysis_snapshot is not None:self.owner.reopen_analysis()
        else:self.show_panel('time')
    def edit_eye_height(self):
        if self.owner is None:return
        worker=getattr(self.search,'worker',None)
        if (worker is not None and worker.is_alive()) or getattr(self.search,'_fill_running',False):
            QtWidgets.QMessageBox.information(self,'Observer height','Wait for the current search to finish before changing the observer height.')
            return
        old=terrain.EYE_HEIGHT_M
        value,accepted=QtWidgets.QInputDialog.getDouble(self,'Observer height','Global height above the terrain (m):',float(old),float(terrain.MIN_EYE_HEIGHT_M),float(terrain.MAX_EYE_HEIGHT_M),2)
        if not accepted or value==old:return
        self.owner.set_eye_height(value)
        self.update_observer_button()
        if self.search.results:self.search.status.configure(text='Observer height changed. Existing search results use the previous height; run the search again.')
        self.owner.window.Render()
    def sync_zen_availability(self):
        if self.owner is None:return
        self.zen_button.setEnabled(not self.owner._graph_expanded and not self.sidebar.isVisible())
    def update_map_button(self):
        self.map_button.setText('Layers · '+self.owner._map_wanted)
    def populate_maps(self):
        self.map_menu.clear()
        for mode in ('Color','Elevation','Slope'):
            action=self.map_menu.addAction(mode);action.setCheckable(True);action.setChecked(self.owner._map_wanted==mode)
            action.triggered.connect(lambda _checked=False,m=mode:self.select_map(m))
    def select_map(self,mode):
        owner=self.owner
        if owner.coordinate_editing or owner._graph_expanded:return
        owner._map_wanted=owner._horizon_wanted=mode;owner._apply_loaded_maps();owner._schedule_map_generation();self.update_map_button();owner.window.Render()
    def set_overlay(self,key,value):
        self.owner._overlay_settings[key]=value
        if key=='map_legend' and value:self.owner._map_legend_open=True
        self.sync_reopen_buttons();self.owner.window.Render()
    def populate_controls(self):
        menu=self.controls_menu;menu.clear();owner=self.owner
        def command(label,callback):
            action=menu.addAction(label);action.triggered.connect(lambda _checked=False:callback())
        def option(label,value,callback):
            action=menu.addAction(label);action.setCheckable(True);action.setChecked(bool(value));action.toggled.connect(callback)
        for label,callback in [('Coordinates…  G',owner.begin_coordinate_entry),('Manage points…  P',owner.begin_points_entry),('Add selected point  A',owner.add_current_point),('Analyze searched point  I',owner.analyze_selected_search_point),('Terrain view  V',owner.toggle_terrain_view),('Reset view  R',self.reset_view),('Fullscreen  F',owner.toggle_fullscreen)]:command(label,callback)
        menu.addSeparator();option('Coordinate grid  L',owner._grid_enabled,self.set_grid)
        for key,label in [('horizon_inset','Horizon graph'),('horizon_outline','Horizon outline'),('map_legend','Map legend'),('grid_caption','Grid information'),('points_caption','Selected points information')]:
            option(label,owner._overlay_settings[key],lambda value,k=key:self.set_overlay(k,value))
    def set_grid(self,value):
        self.owner._grid_enabled=value;self.owner.window.Render()
    def reset_view(self):
        if self.owner._terrain_view:self.owner.toggle_terrain_view(render=False)
        self.owner.reset_camera();self.owner.window.Render()
    def close_panel(self):
        if self.current_panel:self._closed_panels.add(self.current_panel)
        self.sidebar.hide();self.current_panel=None
        for button in self.nav_buttons.values():button.setChecked(False)
        self.sync_zen_availability();self.sync_reopen_buttons()
    def toggle_zen(self):
        if not self._zen and (self.owner._graph_expanded or self.sidebar.isVisible()):return
        self._zen=not self._zen;self.header.setVisible(not self._zen);self.zen_hint.setVisible(self._zen)
        if self._zen:self._zen_panel=self.current_panel;self.sidebar.hide()
        elif getattr(self,'_zen_panel',None):self.show_panel(self._zen_panel)
        self.owner.set_zen(self._zen);self.vtk_widget.setFocus()
    def exit_zen(self):
        if self._zen:self.toggle_zen()
    def show_panel(self,name):
        if name not in self.pages:return
        if self._zen:self.toggle_zen()
        self._closed_panels.discard(name)
        self.stack.setCurrentWidget(self.pages[name]);self.sidebar.show();self.current_panel=name
        for key,button in self.nav_buttons.items():button.setChecked(key==name)
        sizes=self.splitter.sizes()
        if not sizes or sizes[0]<200:self.splitter.setSizes([420,max(600,self.width()-420)])
        self.sync_zen_availability();self.sync_reopen_buttons()
    def toggle_panel(self,name):
        if self.sidebar.isVisible() and self.current_panel==name:
            self.close_panel()
        else:self.show_panel(name)
    def closeEvent(self,event):
        if self.owner is not None:
            self.owner._sky_stop.set()
            self.search.close()
            if self.owner._coordinate_root is not None:
                self.owner._coordinate_root.destroy();self.owner._coordinate_root=None
        event.accept()


def create_application():
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    font=QtGui.QFont('Courier New');font.setStyleHint(QtGui.QFont.Monospace);font.setPointSize(10);app.setFont(font)
    app.setStyleSheet(STYLE)
    return app
