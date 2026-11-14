from astroquery.jplhorizons import Horizons
"""
Alt in km!!!
lon and lat in deg
start, stop format: YYYY-MM-DD [HH:MM:SS]
step in n[y|d|m|s|h]
body: Earth(399), Sun(10)
"""
def get_angle(lon, lat, alt, t_start, t_stop, t_step, body):
    if body == "Earth":
        body = 399 # 399 is code for Earth in horizon, didnt work with name (ValueError: Ambiguous target name; provide unique id)
    obj = Horizons(
        id=body, # sun id
        id_type=None,
        location={"lon": lon, "lat": lat, "elevation": alt, "body": 301},  # lon/lat in deg, altitude in km, 301 - moon id
        epochs={"start": t_start, "stop": t_stop, "step": f"{t_step}"},
    )
    eph = obj.ephemerides(quantities="4", refraction=False) #quantitis = "4" outputs azimuth and elevation in deg.
    return eph["datetime_str", "AZ", "EL"].to_pandas()


# get_angle(0.0,0.0,0.0,"2026-1-1","2026-1-10","1d","Earth")
