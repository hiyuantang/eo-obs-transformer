# gee_utils.py (Complete Code for Image Patch Export)

import ee
import time
import math
import pandas as pd
import os # Added for path joining

# It's recommended to Initialize GEE only ONCE in your main script (get_data.py)
# with the correct project ID, and remove/comment out this block.
try:
    # Replace with your GEE Project ID if needed, otherwise rely on global init
    # Ensure this project ID matches the one used in get_data.py if you keep this block
    GEE_PROJECT_ID = 'animated-way-451621-i3' # Or get from config
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project=GEE_PROJECT_ID)
    print("GEE Initialized in gee_utils (WARNING: recommend initializing only in get_data.py).")
except Exception as e:
    print(f"GEE Initialization failed in gee_utils: {e}. Assuming already initialized elsewhere.")


# Constants based on the paper
KERNEL_SIZE = 224  # Pixel dimensions of the square patch (224x224)
# KERNEL_SIZE = 384  # Pixel dimensions of the square patch (384x384)
PIXEL_SCALE = 30   # Resolution in meters (30m for Landsat)
# Kernel dimensions string for export
KERNEL_DIMS = f"{KERNEL_SIZE}x{KERNEL_SIZE}"

# --- Landsat Data ---
def get_landsat_collection(start_date, end_date, region, landsat_bands):
    """
    Creates a harmonized Landsat 5, 7, 8 SR collection with cloud masking.
    Uses Collection 2, Level 2 data. Selects and renames bands.
    """
    # --- Cloud Masking & Scaling Functions (Landsat C2 L2) ---
    def scale_and_mask_l457(image):
        qa = image.select('QA_PIXEL')
        cloud_shadow_bit_mask = (1 << 3); clouds_bit_mask = (1 << 4)
        mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B6').multiply(0.00341802).add(149.0) # Kelvin
        band_map = {'BLUE': 'SR_B1', 'GREEN': 'SR_B2', 'RED': 'SR_B3', 'NIR': 'SR_B4',
                    'SWIR1': 'SR_B5', 'SWIR2': 'SR_B7', 'TEMP1': 'ST_B6'} # Use TEMP1
        c2_bands = [band_map[b] for b in landsat_bands if b in band_map]
        return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True) \
                    .updateMask(mask).select(c2_bands, landsat_bands) \
                    .copyProperties(image, ["system:time_start"])

    def scale_and_mask_l8(image):
        qa = image.select('QA_PIXEL')
        cloud_shadow_bit_mask = (1 << 3); clouds_bit_mask = (1 << 4)
        mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B10').multiply(0.00341802).add(149.0) # Kelvin
        band_map = {'BLUE': 'SR_B2', 'GREEN': 'SR_B3', 'RED': 'SR_B4', 'NIR': 'SR_B5',
                    'SWIR1': 'SR_B6', 'SWIR2': 'SR_B7', 'TEMP1': 'ST_B10'} # Use TEMP1
        c2_bands = [band_map[b] for b in landsat_bands if b in band_map]
        return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True) \
                    .updateMask(mask).select(c2_bands, landsat_bands) \
                    .copyProperties(image, ["system:time_start"])

    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(region).filterDate(start_date, end_date).map(scale_and_mask_l8)
    l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(region).filterDate(start_date, end_date).map(scale_and_mask_l457)
    l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(region).filterDate(start_date, end_date).map(scale_and_mask_l457)
    collection = ee.ImageCollection(l5.merge(l7).merge(l8)).sort('system:time_start')
    return collection.select(landsat_bands)


# --- Nighttime Lights (NL) Data ---
def get_dmsp_nl_collection(start_year, end_year, region):
    """Gets DMSP-OLS Stable Lights"""
    dmsp_end_year_cutoff = 2011 # DMSP generally useful up to 2011/2012
    actual_end_year = min(end_year, dmsp_end_year_cutoff)
    if start_year > actual_end_year: return ee.ImageCollection([])
    dmsp_start = f'{start_year}-01-01'; dmsp_end = f'{actual_end_year}-12-31'
    # Using stable lights, check GEE catalog for alternatives if needed
    nl_coll = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS') \
                .filterBounds(region).filterDate(dmsp_start, dmsp_end).select('stable_lights')
    # Rename and ensure float type for consistency
    return nl_coll.map(lambda img: img.rename('NL').float().copyProperties(img, ["system:time_start"]))

def get_viirs_nl_collection(start_year, end_year, region):
    """Gets VIIRS monthly nighttime lights data."""
    viirs_start_year_cutoff = 2012 # VIIRS available from ~2012
    actual_start_year = max(start_year, viirs_start_year_cutoff)
    if actual_start_year > end_year: return ee.ImageCollection([])
    viirs_start = f'{actual_start_year}-01-01'; viirs_end = f'{end_year}-12-31'
    nl_coll = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG') \
                .filterBounds(region).filterDate(viirs_start, viirs_end).select('avg_rad')
    # Rename and ensure float type
    return nl_coll.map(lambda img: img.rename('NL').float().copyProperties(img, ["system:time_start"]))

# --- Temporal Compositing ---
def create_temporal_composites(collection, start_year, end_year, span_length, reducer, band_names):
    """
    Creates temporal composites (e.g., medians) over specified year spans.
    MODIFIED: Sets system:index for cleaner band names after toBands().
    """
    composites = ee.ImageCollection([])
    num_bands = len(band_names)
    for year in range(start_year, end_year + 1, span_length):
        span_start_yr = year; span_end_yr = min(year + span_length - 1, end_year)
        span_start_date = f'{span_start_yr}-01-01'; span_end_date = f'{span_end_yr}-12-31'
        span_coll = collection.filterDate(span_start_date, span_end_date)
        count = span_coll.size()
        # Create masked image with correct bands if no source data for the period
        default_image = ee.Image(ee.Array(ee.List.repeat(0, num_bands))).arrayProject([0]).arrayFlatten([band_names]).selfMask()
        composite = ee.Algorithms.If(count.gt(0), span_coll.reduce(reducer).rename(band_names), default_image)

        # Generate year span strings
        year_span_hyphen = f'{span_start_yr}-{span_end_yr}' # Original format (e.g., 1990-1992)
        year_span_underscore = f'{span_start_yr}_{span_end_yr}' # Format for index (e.g., 1990_1992)

        composite = ee.Image(composite).set({
            'system:time_start': ee.Date(span_start_date).millis(),
            'system:time_end': ee.Date(span_end_date).millis(),
            'composite_year_start': span_start_yr,
            'year_span': year_span_hyphen, # Keep original hyphen format if needed
            'system:index': year_span_underscore # *** SET SYSTEM INDEX for toBands() ***
        })
        composites = composites.merge(ee.ImageCollection([composite]))
    return composites

def create_nightlight_composites(start_year, end_year, span_length, region, reducer=ee.Reducer.median()):
    """Creates combined DMSP and VIIRS 3-year composites for Nighttime Lights."""
    print(f"Creating NL composites: DMSP (pre-2012), VIIRS (2012-onwards)")
    dmsp_coll = get_dmsp_nl_collection(start_year, end_year, region)
    viirs_coll = get_viirs_nl_collection(start_year, end_year, region)
    # Merge raw collections before compositing
    combined_nl_coll = ee.ImageCollection(dmsp_coll.merge(viirs_coll))
    nl_size = combined_nl_coll.size().getInfo() # Check size after merge
    if nl_size == 0:
        print("Warning: Combined NL collection is empty before compositing.")
        return ee.ImageCollection([])
    else: print(f"Combined NL collection size: {nl_size}")
    # Composite the merged collection
    nl_composites = create_temporal_composites(combined_nl_coll, start_year, end_year,
                                             span_length, reducer, ['NL']) # Expect 'NL' band
    print(f"NL composites created: {nl_composites.size().getInfo()} images.")
    return nl_composites


# --- Feature Extraction and Export (MODIFIED for Image Patch Export) ---

def export_images(df, country, year, export_folder, export, bucket, ms_bands, include_nl,
                  start_year, end_year, span_length, chunk_size, already_in_bucket=None):
    """
    Exports satellite image PATCHES (Landsat + optional Nightlights) for given
    locations as TFRecords using Export.image.toCloudStorage.

    Args:
        df (pd.DataFrame): DataFrame subset for the specific survey (e.g., test_df or full survey_df).
        country (str): Country name (used for metadata/path).
        year (int): Survey year (used for metadata/path).
        export_folder (str): Base GCS folder path within the bucket or Drive folder name.
        export (str): Export destination ('gcs' or 'drive').
        bucket (str): GCS bucket name (if export_dest is 'gcs').
        ms_bands (list): List of Landsat band names (e.g., ['BLUE', ..., 'TEMP1']).
        include_nl (bool): Whether to include nighttime lights band.
        start_year (int): Start year for image time series.
        end_year (int): End year for image time series.
        span_length (int): Number of years per temporal composite.
        chunk_size (int): Max number of export tasks to START in one call to this function.
        already_in_bucket (list, optional): List of location indices (0 to N-1 for this df)
                                            to SKIP exporting.

    Returns:
        dict: Dictionary of {task_id: ee.batch.Task} for started export tasks FOR THIS RUN.
    """
    if already_in_bucket is None:
        already_in_bucket = []

    tasks = {}

    # Use the DataFrame passed into the function directly
    survey_df = df.reset_index(drop=True) # Ensure index is clean 0 to N-1

    # # Optional check (can be removed if input df is guaranteed correct)
    # if not all(survey_df['country'] == country) or not all(survey_df['year'] == year):
    #    print(f"Warning: DataFrame passed to export_images for {country} {year} contains unexpected rows.")
    #    # Filter to be safe? Or rely on caller providing correct df.
    #    # survey_df = survey_df[(survey_df['country'] == country) & (survey_df['year'] == year)].reset_index(drop=True)


    if survey_df.empty:
        print(f"Warning: Empty DataFrame passed for {country} {year}. Skipping export.")
        return tasks

    # Ensure unique ID column exists or create one using the index of the PASSED df
    if 'uid' not in survey_df.columns:
        # Use padding for consistent sorting if needed
        survey_df['uid'] = [f"{country}_{year}_{i:05d}" for i in range(len(survey_df))]
        print("Warning: No 'uid' column found. Creating unique IDs like 'country_year_00000'.")

    num_locations = len(survey_df)
    print(f"Processing {country} {year}: {num_locations} locations from input DataFrame.")
    print(f"Exporting ONE TFRecord per location.")
    print(f"Skipping location indices based on already_in_bucket: {already_in_bucket}")
    print(f"Time period: {start_year}-{end_year}, Composites: {span_length}-year spans.")

    # --- Create Composite Image Collection (Once per survey) ---
    # Get overall region covering points in the input df for initial filtering
    points = [ee.Geometry.Point(row['lon'], row['lat']) for idx, row in survey_df.iterrows()] # Use lat/lon
    if not points:
        print(f"Error: No valid points found for {country} {year} in input DataFrame.")
        return {}
    # Use a larger buffer radius just for initial collection filtering
    buffer_radius_approx = KERNEL_SIZE * PIXEL_SCALE * 1.5
    overall_region = ee.FeatureCollection(points).geometry().bounds().buffer(buffer_radius_approx)

    # 1. Get Landsat Collection
    landsat_coll = get_landsat_collection(f'{start_year}-01-01', f'{end_year}-12-31', overall_region, ms_bands)

    # 2. Create Landsat Composites
    landsat_composites = create_temporal_composites(
        landsat_coll, start_year, end_year, span_length, ee.Reducer.median(), ms_bands
    )

    # Check if Landsat composites were generated
    if landsat_composites.size().getInfo() == 0:
         print(f"Error: No Landsat composite images generated for {country} {year}. Cannot export.")
         return {}

    # 3. Create NL Composites (if requested)
    nl_composites = None # Initialize
    all_bands = list(ms_bands) # Bands expected in the final stack
    if include_nl:
        print("Including Nighttime Lights...")
        nl_composites = create_nightlight_composites(
            start_year, end_year, span_length, overall_region, reducer=ee.Reducer.median() # Or mean()
        )
        if nl_composites.size().getInfo() > 0:
            print("NL Composites successfully created.")
            # Use 'NL' band name. If downstream needs 'NIGHTLIGHTS', change here or there.
            all_bands.append('NL')
            # nl_composites = nl_composites.map(lambda img: img.rename('NIGHTLIGHTS'))
            # all_bands.append('NIGHTLIGHTS')
        else:
             print("Warning: No NL composites generated. Proceeding with Landsat only.")
             include_nl = False # Update flag

    # --- Convert Image Collections to Multi-Band Image Stacks (Simplified Logic) ---

    # Define the function to add time suffix to band names
    def add_time_suffix(img):
        img = ee.Image(img) # Cast to image just in case
        year_span = ee.String(img.get('year_span'))
        time_suffix = year_span.replace('-', '_', 'g') # e.g., 1990_1992
        # Function needs to be defined within the scope or passed arguments
        def rename_band(b):
             return ee.String(b).cat('_').cat(time_suffix)
        # Check if image has bands before trying to map
        has_bands = img.bandNames().size().gt(0)
        return ee.Algorithms.If(has_bands,
                                img.rename(img.bandNames().map(rename_band)),
                                img) # Return original image if it has no bands (shouldn't happen with default_image)

    # Apply suffix and stack Landsat bands
    landsat_stacked = landsat_composites.map(add_time_suffix).toBands()
    print(f"Created Landsat stacked image with {landsat_stacked.bandNames().size().getInfo()} bands.")

    stacked_image = landsat_stacked # Initialize final stacked image

    # Apply suffix, stack, and add NL bands if included and available
    if include_nl and nl_composites is not None and nl_composites.size().getInfo() > 0:
        nl_stacked = nl_composites.map(add_time_suffix).toBands()
        nl_band_count = nl_stacked.bandNames().size().getInfo()
        if nl_band_count > 0:
            print(f"Created NL stacked image with {nl_band_count} bands.")
            stacked_image = stacked_image.addBands(nl_stacked)
        else:
             print("Warning: NL stacking skipped as nl_stacked image has 0 bands.")
             if 'NL' in all_bands: all_bands.remove('NL')
             # if 'NIGHTLIGHTS' in all_bands: all_bands.remove('NIGHTLIGHTS')
    elif include_nl:
        print("NL stacking skipped as no NL composites were initially generated.")
        if 'NL' in all_bands: all_bands.remove('NL')
        # if 'NIGHTLIGHTS' in all_bands: all_bands.remove('NIGHTLIGHTS')


    # Final check on the combined stacked image
    final_band_count = stacked_image.bandNames().size().getInfo()
    if final_band_count == 0:
        print(f"Error: Final stacked image has 0 bands for {country} {year}. Cannot export.")
        return {}
    print(f"Created final stacked image with {final_band_count} total bands.")


    # --- Export Loop (One Task Per Location) ---
    tasks_started_count = 0
    # Use chunk_size to limit tasks started *in this specific function call*
    max_tasks_to_start = chunk_size

    for index, row in survey_df.iterrows():
        # Check if this location index (relative to the input df) should be skipped
        if index in already_in_bucket:
            # print(f"Skipping location index {index} (already in bucket/list).")
            continue

        # Stop submitting tasks if the limit for this run is reached
        if tasks_started_count >= max_tasks_to_start:
             print(f"Reached max tasks ({max_tasks_to_start}) to start for this run. Submit remaining later.")
             break

        location_uid = row['uid']
        print(f"--- Processing Location Index: {index} (UID: {location_uid}) ---")

        point = ee.Geometry.Point(row['lon'], row['lat']) # Use lon/lat from CSV
        # Define the precise 224x224 export region centered on the point
        # Use buffer().bounds() method for a square region centered on point
        export_region = point.buffer(KERNEL_SIZE * PIXEL_SCALE / math.sqrt(2)).bounds() # Buffer by half-diagonal -> bounds gives enclosing square

        # Define task description and filename
        task_description = f"Export_{location_uid}" # Must be unique across all exports eventually
        # Store files in subdirectories per survey: FOLDER/COUNTRY_YEAR/UID.tfrecord.gz
        file_prefix = f"{export_folder}/{country}_{year}/{location_uid}" # GCS path format

        print(f"Starting export task: {task_description} for region: {export_region.getInfo()['coordinates']}") # Log region

        export_params = {
            'image': stacked_image.toFloat(), # Ensure float32 output, GEE default is often double
            'description': task_description,
            'scale': PIXEL_SCALE,
            'region': export_region,
            'fileFormat': 'TFRecord',
            # Specify patch dimensions and output format details
            'formatOptions': {
                'patchDimensions': [KERNEL_SIZE, KERNEL_SIZE], # 224x224
                'kernelSize': [0, 0], # No overlap for single patch export
                'compressed': True, # Output .tfrecord.gz
                'maxFileSize': 100 * 1024 * 1024, # Optional: Limit file size (e.g., 100MB)
                # 'defaultValue': 0 # Optional: value for masked pixels (use with caution)
                # If properties needed in TFRecord, use Export.table with getRegion/sampleRectangle
                # For image patches, metadata usually stored separately or in filename.
            }
        }

        if export.lower() == 'gcs':
            if not bucket:
                print(f"ERROR: Bucket name missing for GCS export (Location {index}). Skipping task.")
                continue
            export_params['bucket'] = bucket
            export_params['fileNamePrefix'] = file_prefix
            task = ee.batch.Export.image.toCloudStorage(**export_params)
        elif export.lower() == 'drive':
             # Create survey subfolder in Drive export folder
            drive_folder = f"{export_folder}/{country}_{year}"
            export_params['folder'] = drive_folder
            export_params['fileNamePrefix'] = location_uid # Filename in Drive folder
            task = ee.batch.Export.image.toDrive(**export_params)
        else:
            print(f"Export destination '{export}' not supported (Location {index}).")
            continue

        try:
            task.start()
            tasks[task.id] = task
            tasks_started_count += 1
            print(f"Task {task.id} started for location index {index} (UID: {location_uid}).")
            # Optional: Brief sleep to avoid client-side rate limits if submitting many tasks
            # time.sleep(0.5)
        except ee.EEException as e:
            print(f"ERROR starting task for location index {index} (UID: {location_uid}): {e}")
            # Consider logging failed task submissions

    print(f"Finished submitting tasks for {country} {year}. Total tasks started in this run: {tasks_started_count}")
    return tasks


# --- Wait on Tasks ---
# (wait_on_tasks function remains unchanged - keep the robust version from previous steps)
def wait_on_tasks(tasks, poll_interval=60):
    """
    Waits for a list or dictionary of GEE tasks to complete. Provides status updates.
    """
    print(f"Waiting on {len(tasks)} tasks...")
    tasks_list = list(tasks.values()) if isinstance(tasks, dict) else tasks
    if not tasks_list:
        print("No tasks to wait for.")
        return

    running_tasks = list(tasks_list) # Copy list to modify while iterating

    while running_tasks:
        finished_tasks = []
        status_counts = {'RUNNING': 0, 'COMPLETED': 0, 'FAILED': 0, 'CANCELLED': 0, 'READY': 0, 'SUBMITTED': 0, 'UNKNOWN': 0}
        all_done = True

        for task in running_tasks:
            try:
                status = task.status()
                state = status.get('state', 'UNKNOWN')
                status_counts[state] = status_counts.get(state, 0) + 1

                if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    finished_tasks.append(task)
                    if state == 'FAILED':
                         print(f"WARNING: Task {task.id} ({status.get('description', 'N/A')}) FAILED. Error: {status.get('error_message', 'Unknown error')}")
                    elif state == 'CANCELLED':
                         print(f"INFO: Task {task.id} ({status.get('description', 'N/A')}) was CANCELLED.")
                elif state in ['READY', 'SUBMITTED', 'RUNNING']:
                    all_done = False # Task is still active
                else: # UNKNOWN state or other
                    all_done = False # Assume not done unless proven otherwise
            except Exception as e:
                print(f"Error checking status for task {task.id}: {e}")
                status_counts['UNKNOWN'] += 1
                # Keep task in running_tasks on status check error, assume it might recover
                all_done = False

        # Remove finished tasks from the list we are actively checking
        for task in finished_tasks:
            if task in running_tasks:
                try: running_tasks.remove(task)
                except ValueError: pass # Task might have already been removed

        # Print status update
        status_line = ', '.join([f"{state}: {count}" for state, count in status_counts.items() if count > 0])
        print(f"Task Status ({time.strftime('%Y-%m-%d %H:%M:%S')}): {status_line}. Remaining: {len(running_tasks)}")

        if all_done or not running_tasks:
            break # Exit loop if all tasks finished or list is empty

        time.sleep(poll_interval)

    print("All monitored tasks have finished or encountered errors.")
    # Final summary of originally submitted tasks
    final_statuses = {}; total = 0; failed_tasks_summary = []
    for task_obj in tasks_list: # Iterate original list for final summary
         try:
            status = task_obj.status() # Re-check status for final summary
            state = status.get('state', 'UNKNOWN')
            final_statuses[state] = final_statuses.get(state, 0) + 1
            total += 1
            if state == 'FAILED':
                 failed_tasks_summary.append(status.get('description', task_obj.id))
         except Exception:
             final_statuses['UNKNOWN'] = final_statuses.get('UNKNOWN', 0) + 1
             total += 1
             failed_tasks_summary.append(f"{task_obj.id} (final status check failed)")
    print(f"Final Summary ({total} tasks submitted in this run): ", ', '.join([f"{state}: {count}" for state, count in final_statuses.items()]))
    if failed_tasks_summary:
         print(f"Failed Tasks Summary: {', '.join(failed_tasks_summary)}")