"""Coarse, cancellable lunar surface search for terrain-aware visibility.

Run via S in moon.py. Skyfield is used locally. Percentages are estimated by
trapezoidal integration of sampled visibility (partly visible counts; Earth
occulting the Sun is included). No slope requirement is applied.
"""
import csv
import queue
import threading
import time
from datetime import timedelta

import numpy as np
import sky
import terrain

LUNAR_DAY_DAYS = 29.530588861
MAX_GRID_POINTS = 100_000
MAX_DISPLAY_RESULTS = 500


def grid_points(lat_min, lat_max, lon_min, lon_max, spacing):
    values = [float(lat_min), float(lat_max), float(lon_min), float(lon_max), float(spacing)]
    if not np.isfinite(values).all():
        raise ValueError('Grid values must be finite.')
    lat_min, lat_max, lon_min, lon_max, spacing = values
    if not (-90 <= lat_min <= lat_max <= 90):
        raise ValueError('Latitude range must be ordered and within -90 to 90 degrees.')
    if not (-180 <= lon_min <= lon_max <= 180):
        raise ValueError('Longitude range must be ordered and within -180 to 180 degrees.')
    if not 0.1 <= spacing <= 180:
        raise ValueError('Grid spacing must be between 0.1 and 180 degrees.')
    count_lat = int(np.floor((lat_max - lat_min) / spacing + 1e-9)) + 1
    count_lon = int(np.floor((lon_max - lon_min) / spacing + 1e-9)) + 1
    if count_lat * count_lon > MAX_GRID_POINTS + count_lon * 2:
        raise ValueError(f'Grid is too large; increase spacing or narrow the region (maximum {MAX_GRID_POINTS:,} points).')
    lats = lat_min + np.arange(count_lat) * spacing
    lons = lon_min + np.arange(count_lon) * spacing
    if len(lons) > 1 and np.isclose(lons[-1] - lons[0], 360.0, atol=1e-9, rtol=0):
        lons = lons[:-1]
    points = []
    for lat in lats:
        if np.isclose(abs(lat), 90.0, atol=1e-9, rtol=0):
            points.append((float(lat), float(lons[0])))
        else:
            points.extend((float(lat), float(lon)) for lon in lons)
    if len(points) > MAX_GRID_POINTS:
        raise ValueError(f'Grid is too large; maximum {MAX_GRID_POINTS:,} points.')
    return points


def visibility_percent(epochs, mask):
    """Time-weight samples with midpoint crossings; handles a short last step."""
    if len(epochs) < 2:
        raise ValueError('Start and end must be different.')
    values = np.asarray(mask, dtype=float)
    if values.shape != (len(epochs),):
        raise ValueError('Visibility result does not match the time samples.')
    durations = np.array([(b - a).total_seconds() for a, b in zip(epochs[:-1], epochs[1:])])
    if np.any(durations <= 0):
        raise ValueError('Times must be in strictly increasing order.')
    return float(100 * np.dot(durations, (values[:-1] + values[1:]) * 0.5) / durations.sum())


def search(points, epochs, min_sun, min_earth, fetch, cancel=None, progress=None):
    """Return passing sites; stop between sites if cancel is set.

    `fetch(lon, lat, altitude_km, epochs)` returns Sun and Earth Skyfield
    arrays. Errors abort rather than silently classifying failed sites as dark.
    """
    thresholds = np.asarray([min_sun, min_earth], dtype=float)
    if not np.isfinite(thresholds).all() or np.any(thresholds < 0) or np.any(thresholds > 100):
        raise ValueError('Visibility thresholds must be between 0 and 100 percent.')
    if len(epochs) < 2 or epochs[-1] <= epochs[0]:
        raise ValueError('End time must be later than start time.')
    results = []
    start = time.monotonic()
    completed = 0
    for index, (lat, lon) in enumerate(points):
        if cancel is not None and cancel.is_set():
            break
        if progress:
            elapsed = time.monotonic() - start
            progress(index, len(points), len(results), elapsed, lat, lon)
        try:
            horizon = terrain.compute_horizon_360(lat, lon)[0]
            if cancel is not None and cancel.is_set():
                break
            altitude = (float(terrain.sample_dem_bilinear(lat, lon)) + 2.0) / 1000.0
            positions = fetch(lon, lat, altitude, epochs)
            if cancel is not None and cancel.is_set():
                break
            classified = sky.classify_sky(positions, horizon)
            sun = visibility_percent(epochs, classified['Sun']['visible'])
            earth = visibility_percent(epochs, classified['Earth']['visible'])
        except Exception as error:
            raise RuntimeError(f'Search failed at latitude {lat:.3f}, longitude {lon:.3f}: {error}') from error
        completed += 1
        if sun >= min_sun and earth >= min_earth:
            results.append({'lat': lat, 'lon': lon, 'sun_percent': sun, 'earth_percent': earth})
    if progress:
        progress(completed,
                 len(points), len(results), time.monotonic() - start, None, None)
    return sorted(results, key=lambda p: (min(p['sun_percent'] - min_sun, p['earth_percent'] - min_earth),
                                           p['sun_percent'] + p['earth_percent']), reverse=True)


def _search_worker(messages, cancel, points, epochs, sun, earth, fetch):
    def progress(done, total, matches, elapsed, lat, lon):
        messages.put(('progress', (done, total, matches, elapsed, lat, lon)))
    try:
        results = search(points, epochs, sun, earth, fetch, cancel, progress)
        messages.put(('done', (results, cancel.is_set())))
    except Exception as error:
        messages.put(('error', str(error)))


class SearchDialog:
    """Tk-only UI; all horizon and ephemeris work stays in a worker thread."""
    def __init__(self, parent, fetch, previous=None, on_close=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.fetch = tk, ttk, fetch
        self.on_close = on_close
        self._closed = False
        self.window = tk.Toplevel(parent)
        self.window.title('Search lunar visibility')
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.geometry('780x720')
        self.messages = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.timer = None
        self.results = []
        self.selected = None
        self.show_matches = False
        self.clear_markers = False
        self.last_parameters = None
        self.finished = False
        self.was_cancelled = False
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill='both', expand=True)
        settings = ttk.Frame(frame)
        settings.pack(fill='x')
        now = sky.utc_now()
        defaults = [
            ('Start UTC', now.strftime('%Y-%m-%d %H:%M:%S')),
            ('End UTC', (now + timedelta(days=LUNAR_DAY_DAYS)).strftime('%Y-%m-%d %H:%M:%S')),
            ('Time step (hours)', '2'), ('Minimum Sun visible (%)', '30'),
            ('Minimum Earth visible (%)', '50'), ('Grid spacing (degrees)', '10'),
            ('Latitude min / max', '-90, 90'), ('Longitude min / max', '-180, 180'),
        ]
        self.fields = []
        for index, (label, value) in enumerate(defaults):
            row, col = divmod(index, 2)
            ttk.Label(settings, text=label).grid(row=row, column=col * 2, sticky='w', padx=(0, 5), pady=3)
            entry = ttk.Entry(settings, width=23)
            entry.insert(0, value)
            entry.grid(row=row, column=col * 2 + 1, sticky='ew', padx=(0, 12), pady=3)
            self.fields.append(entry)
        ttk.Label(frame, text='Skyfield; percentages include partial visibility and Earth blocking the Sun.\n'
                              'Coarse samples are approximate; one lunar day is about 29.53 Earth days.',
                  wraplength=720).pack(anchor='w', pady=(8, 4))
        controls = ttk.Frame(frame)
        controls.pack(fill='x')
        self.start_button = ttk.Button(controls, text='Start search', command=self.start)
        self.start_button.pack(side='left')
        self.cancel_button = ttk.Button(controls, text='Cancel search', command=self.cancel.set, state='disabled')
        self.cancel_button.pack(side='left', padx=6)
        self.progress = ttk.Progressbar(frame, mode='determinate')
        self.progress.pack(fill='x', pady=(8, 3))
        self.status = ttk.Label(frame, text='Choose criteria and start the search.', wraplength=720)
        self.status.pack(anchor='w')
        columns = ('lat', 'lon', 'sun', 'earth')
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill='both', expand=True, pady=8)
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=14, selectmode='browse')
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        for key, label in zip(columns, ('Latitude', 'Longitude', 'Sun visible %', 'Earth visible %')):
            self.table.heading(key, text=label)
            self.table.column(key, width=150, anchor='center')
        self.table.pack(side='left', fill='both', expand=True)
        self.table.bind('<Double-1>', self.view_selected)
        buttons = ttk.Frame(frame)
        buttons.pack(fill='x')
        self.view_button = ttk.Button(buttons, text='Show selected on globe', command=self.view_selected, state='disabled')
        self.view_button.pack(side='left')
        self.marks_button = ttk.Button(buttons, text='Show matches on globe', command=self.show_all, state='disabled')
        self.marks_button.pack(side='left', padx=5)
        self.export_button = ttk.Button(buttons, text='Save all results as CSV', command=self.export, state='disabled')
        self.export_button.pack(side='left', padx=5)
        ttk.Button(buttons, text='Close', command=self.close).pack(side='right')
        ttk.Button(buttons, text='Clear markers', command=self.clear).pack(side='right', padx=5)
        self.window.bind('<Escape>', lambda event: self.close())
        if previous is not None:
            self.restore(previous)
        # The Tk root is withdrawn and is not the VTK window. Making this
        # dialog transient or grabbing it can leave it hidden behind VTK.
        self.window.deiconify()
        self.window.update_idletasks()
        self.window.lift()
        self.window.focus_force()
        self.poll()

    @staticmethod
    def _pair(text):
        values = text.replace(';', ',').split(',')
        if len(values) != 2:
            raise ValueError('Enter each coordinate range as min, max.')
        return tuple(map(float, values))

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            start, stop = sky.parse_utc(self.fields[0].get()), sky.parse_utc(self.fields[1].get())
            step_hours, sun, earth, spacing = (float(self.fields[i].get()) for i in range(2, 6))
            lat_min, lat_max = self._pair(self.fields[6].get())
            lon_min, lon_max = self._pair(self.fields[7].get())
            points = grid_points(lat_min, lat_max, lon_min, lon_max, spacing)
            epochs = sky.time_samples(start, stop, step_hours * 60)
            if len(epochs) < 2:
                raise ValueError('End time must be later than start time.')
            if not np.isfinite([sun, earth]).all() or not (0 <= sun <= 100 and 0 <= earth <= 100):
                raise ValueError('Visibility thresholds must be between 0 and 100 percent.')
        except (ValueError, OverflowError) as error:
            self.status.configure(text=str(error))
            return
        self.cancel.clear()
        self.results = []
        self.selected = None
        self.show_matches = False
        self.finished = False
        self.last_parameters = (start, stop, step_hours, sun, earth, spacing, lat_min, lat_max, lon_min, lon_max)
        self.table.delete(*self.table.get_children())
        for button in (self.view_button, self.marks_button, self.export_button):
            button.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.progress.configure(maximum=len(points), value=0)
        self.status.configure(text=f'Preparing {len(points):,} points, {len(epochs):,} time samples per point...')
        self.worker = threading.Thread(target=_search_worker,
            args=(self.messages, self.cancel, points, epochs, sun, earth, self.fetch), daemon=True)
        self.worker.start()

    def poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == 'progress':
                    done, total, matches, elapsed, lat, lon = value
                    self.progress.configure(value=done)
                    eta = f' | estimated {elapsed / done * (total - done) / 60:.1f} min left' if done else ''
                    where = f' | {lat:+.2f}, {lon:+.2f}' if lat is not None else ''
                    self.status.configure(text=f'{done:,}/{total:,} points | {matches:,} matches | {elapsed / 60:.1f} min elapsed{eta}{where}')
                elif kind == 'done':
                    self.results, cancelled = value
                    self.finished = True
                    self.was_cancelled = cancelled
                    self.populate_results('Cancelled' if cancelled else 'Complete')
                    self.start_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    if self.results:
                        for button in (self.view_button, self.marks_button, self.export_button):
                            button.configure(state='normal')
                else:
                    self.status.configure(text=value)
                    self.start_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
        except queue.Empty:
            pass
        self.timer = self.window.after(100, self.poll)

    def populate_results(self, label):
        self.table.delete(*self.table.get_children())
        for index, row in enumerate(self.results[:MAX_DISPLAY_RESULTS]):
            self.table.insert('', 'end', iid=str(index), values=(f"{row['lat']:+.3f}", f"{row['lon']:+.3f}",
                                                                f"{row['sun_percent']:.1f}", f"{row['earth_percent']:.1f}"))
        suffix = f' Showing first {MAX_DISPLAY_RESULTS}; CSV contains all.' if len(self.results) > MAX_DISPLAY_RESULTS else ''
        self.status.configure(text=f"{label}: {len(self.results):,} matches.{suffix}")
        if self.results:
            for button in (self.view_button, self.marks_button, self.export_button):
                button.configure(state='normal')

    def restore(self, previous):
        self.results = previous['results']
        self.last_parameters = previous['parameters']
        self.was_cancelled = previous['cancelled']
        self.finished = True
        start, stop, step, sun, earth, spacing, lat_min, lat_max, lon_min, lon_max = self.last_parameters
        values = (start.strftime('%Y-%m-%d %H:%M:%S'), stop.strftime('%Y-%m-%d %H:%M:%S'),
                  step, sun, earth, spacing, f'{lat_min:g}, {lat_max:g}', f'{lon_min:g}, {lon_max:g}')
        for field, value in zip(self.fields, values):
            field.delete(0, 'end')
            field.insert(0, str(value))
        self.populate_results('Previous partial search' if self.was_cancelled else 'Previous search')

    def view_selected(self, event=None):
        selected = self.table.selection()
        if selected:
            self.selected = self.results[int(selected[0])]
            self.show_matches = True
            self.close()

    def show_all(self):
        self.show_matches = True
        self.close()

    def clear(self):
        self.clear_markers = True
        self.close()

    def export(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension='.csv',
                                            filetypes=[('CSV files', '*.csv')])
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as output:
                writer = csv.writer(output)
                writer.writerow(('latitude_deg', 'longitude_deg', 'sun_visible_percent', 'earth_visible_percent',
                                 'start_utc', 'end_utc', 'step_hours', 'min_sun_percent', 'min_earth_percent',
                                 'grid_spacing_deg', 'latitude_min', 'latitude_max', 'longitude_min', 'longitude_max',
                                 'partial_search'))
                start, stop, step, sun, earth, spacing, lat_min, lat_max, lon_min, lon_max = self.last_parameters
                for row in self.results:
                    writer.writerow((row['lat'], row['lon'], row['sun_percent'], row['earth_percent'],
                                     start.isoformat(), stop.isoformat(), step, sun, earth, spacing,
                                     lat_min, lat_max, lon_min, lon_max, self.was_cancelled))
            self.status.configure(text=f'Saved {len(self.results):,} results to {path}')
        except OSError as error:
            self.status.configure(text=f'Could not save results: {error}')

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.cancel.set()
        if self.timer is not None:
            try:
                self.window.after_cancel(self.timer)
            except self.tk.TclError:
                pass
        self.window.destroy()
        if self.on_close is not None:
            self.on_close(self)
