import ee
import os
from google.oauth2 import service_account
import json

# Function to initialize Earth Engine with service account
def initialize_ee():
    try:
        # Path to your service account JSON file
        SERVICE_ACCOUNT_KEY = 'ee-thukupeter487soknotproject-0e5f73f036df.json'
        
        # Read credentials from JSON file
        credentials = ee.ServiceAccountCredentials(
            email=None,
            key_file=SERVICE_ACCOUNT_KEY
        )
        
        # Initialize Earth Engine
        ee.Initialize(credentials)
        
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        raise

# Modified version of your imports and initialization
import streamlit as st
import datetime
import geopandas as gpd
import geemap
from shapely.geometry import box, mapping, Point
import folium
import zipfile
from pyproj import Transformer
from fpdf import FPDF
import io
from streamlit_geolocation import streamlit_geolocation
import osmnx as ox
import ipyleaflet
import networkx as nx
import tempfile

# Initialize EE at the start of your app
initialize_ee()

def load_kenyan_counties():
    return gpd.read_file(
        "Counties.shp")

def filter_county_by_name(county, query):
    return county[county['NAME'].str.contains(query, case=False)]

def load_croplands():
    return gpd.read_file(
        "All_Coffee_Farms.shp"
    )

def filter_cropland_by_county_and_type(croplands, selected_county, selected_cropland):
    county_geometry = selected_county.geometry.iloc[0]
    croplands_within_county = croplands[croplands.intersects(county_geometry)]
    croplands_within_county_and_type = croplands_within_county[croplands_within_county['Bname'] == selected_cropland]

    if croplands_within_county_and_type.empty:
        st.sidebar.warning(f"No {selected_cropland} croplands found within the selected county.")
        return None

    croplands_within_county_and_type['geometry'] = croplands_within_county_and_type['geometry'].intersection(
        county_geometry)
    return croplands_within_county_and_type


def create_geemap(OriginCounty, county, croplands_within_county_and_type, forest_image=None, ndvi_image=None, forest_change_image=None, sentinel_image=None, lulc_image=None):
    county_data = county[county['NAME'] == OriginCounty]
    selected_county_geometry = county_data.geometry.iloc[0]

    if croplands_within_county_and_type is None:
        return None

    # Center the map on the county centroid
    m = geemap.Map(center=[selected_county_geometry.centroid.y, selected_county_geometry.centroid.x], zoom=10)

    m.add_gdf(county_data.set_crs('epsg:4326'), layer_name='AOI')
    # Add croplands layer
    m.add_gdf(croplands_within_county_and_type, layer_name='Coffee Farms')


    # Add basemap
    m.add_basemap("SATELLITE")

    # Add NDVI layer if available
    if ndvi_image:
        ndvi_band = ndvi_image.select('NDVI')
        m.addLayer(ndvi_band, {'min': -1, 'max': 1, 'palette': ['red', 'orange', 'green']}, 'NDVI')
        add_ndvi_legend(m)

    # Add Sentinel-2 imagery if available
    if sentinel_image:
        sentinel_vis_params = {'gamma': 1, 'min': 424, 'max': 2030.5, 'opacity': 1, 'bands': ['B4', 'B3', 'B2']}
        m.addLayer(sentinel_image, sentinel_vis_params, 'RGB Sentinel')


    # Add Forested Areas layer if available
    if forest_image:
        forest_non_forest_vis = {
            'band': 'Map',
            'palette': ['#4d9221']
        }
        m.addLayer(forest_image, forest_non_forest_vis, 'Forested Cover')
        add_forest_legend(m)


    # Add Land Use/Land Cover (LULC) layer if available
    if lulc_image:
        lulc_vis_params = {
            'min': 0,
            'max': 8,
            'palette': ['419BDF', '397D49', '88B053', '7A87C6', 'E49635', 'DFC35A', 'C4281B', 'A59B8F', 'B39FE1']
        }
        m.addLayer(lulc_image, lulc_vis_params, 'Land Use/Land Cover')
        add_lulc_legend(m)

    # Add Forest Change layers if available
    if forest_change_image:
        tree_cover_layer, tree_loss_layer = visualize_forest_change(forest_change_image)
        # m.addLayer(tree_cover_layer, {}, 'Tree Cover 2000')
        m.addLayer(tree_loss_layer, {}, 'Tree Loss Year')
        add_forest_change_legend(m)

    return m


def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def calculate_RGB(image):
    rgb = image.select('B4', 'B3', 'B2')
    return rgb

def generate_forestdata(image_ECJRCf):
    forest = image_ECJRCf.select('Map')
    return forest

def generate_forestchange(image_fc):
    forest_change = image_fc.select('treecover2000', 'lossyear')
    return forest_change

def generate_lulc(image_lulc):
    lulc = image_lulc.select('label')
    return lulc

def geodf_to_ee_featurecollection(geodf):
    features = []
    for _, row in geodf.iterrows():
        geom = ee.Geometry(mapping(row['geometry']))
        feature = ee.Feature(geom)
        features.append(feature)
    return ee.FeatureCollection(features)

def add_ndvi_legend(map_obj):
    legend_html = '''
    <div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 250px;
    height: 50px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    ">
    <div style="
    background: linear-gradient(to right, red , orange, green);
    width: 100%;
    height: 25px;
    ">
    </div>
    <div style="text-align: center;">
    <span style="float: left;">-1</span>
    <span>0</span>
    <span style="float: right;">1</span>
    </div>
    <div style="text-align: center;">
    </div>
    </div>
    '''
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def add_forest_legend(map_objf):
    legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 200px; height: 100px;
                    border:2px solid grey; background-color:white; opacity: 0.7; z-index:9999; font-size:14px;
                    ">
        <b>&nbsp;&nbsp;Legend</b><br><br>
        <div style="display: flex; flex-wrap: wrap; width: 300px;"> &nbsp;
            <div style="width: 50%; text-align: left;">
                <i style="background: #4d9221; width: 30px; height: 20px; display: inline-block;"></i> &nbsp;Tree Cover
            </div>

        </div>
        </div>
    '''
    map_objf.get_root().html.add_child(folium.Element(legend_html))

def generate_lulc_data(start_date, end_date):
    # Load the Dynamic World LULC dataset
    lulc_collection = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                       .filterDate(start_date, end_date))

    # Get the median LULC image within the date range
    lulc_image = lulc_collection.median()

    return lulc_image


def visualize_lulc(lulc_image):
    # Define LULC color map
    color_map = {
        0: '#419bdf',  # water
        1: '#397d49',  # trees
        2: '#88b053',  # grass
        3: '#7a87c6',  # flooded_veg
        4: '#e49635',  # crops
        5: '#dfc35a',  # shrub_and_scrub
        6: '#c4281b',  # built
        7: '#a59b8f',  # bare
        8: '#a59b8f'   # snow_ice
    }

    # Create a color palette from the map
    palette = [color_map[i] for i in range(9)]

    # Visualization parameters
    vis_params = {
        'min': 0,
        'max': 8,
        'palette': palette
    }

    # Add a single-band image visualization
    lulc_vis = lulc_image.select('label').visualize(**vis_params)

    return lulc_vis


def generate_forest_data(start_date, end_date):
    # Load the Dynamic World LULC dataset
    forest_collection = (ee.ImageCollection("JRC/GFC2020/V1").first())
                    #    .filterDate(start_date, end_date))

    # Get the median LULC image within the date range
    forest_image = forest_collection

    return forest_image

def generate_forest_change():
    # Load the Dynamic World LULC dataset
    forest_change_collection = (ee.Image('UMD/hansen/global_forest_change_2023_v1_11'))

    # Get the median LULC image within the date range
    forest_change_image = forest_change_collection

    return forest_change_image

def visualize_forest(forest_image):
    # Define LULC color map
    color_map = {

        1: '#4d9221'  # forest
    }

    # Create a color palette from the map
    palette = [color_map[i] for i in range(1)]

    # Visualization parameters
    vis_params = {
        'band': 'Map',
        'palette': palette
    }

    # Add a single-band image visualization
    forest_vis = forest_image.select('Map').visualize(**vis_params)

    return forest_vis

def visualize_forest_change(forest_change_image):
    # Visualization for tree cover 2000
    tree_cover_vis = {
        'bands': ['treecover2000'],
        'min': 0,
        'max': 100,
        'palette': ['black', 'green']
    }
    tree_loss_vis = {
        'bands': ['lossyear'],
        'min': 0,
        'max': 23,
        'palette': ['yellow', 'red']
    }

    tree_cover_layer = forest_change_image.select('treecover2000').visualize(**tree_cover_vis)
    tree_loss_layer = forest_change_image.select('lossyear').visualize(**tree_loss_vis)

    return tree_cover_layer, tree_loss_layer


def calculate_lulc_areas(lulc_image, region):
    lulc_classes = {
        0: 'Water',  # water
        1: 'Trees',  # trees
        2: 'Grass',  # grass
        3: 'Flooded',  # flooded_veg
        4: 'Crops',  # crops
        5: 'Shrubs',  # shrub_and_scrub
        6: 'Built',  # built
        7: 'Bare',  # bare
        8: 'Snow'   # snow_ice
    }

    areas = {}

    for class_value, class_name in lulc_classes.items():
        class_mask = lulc_image.eq(class_value)
        area_m2 = class_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=10,
            maxPixels=1e12
        ).get('label')

    areas[class_name] = ee.Number(area_m2).divide(10000)


def add_lulc_legend(map_obj):
    # Define LULC legend based on Dynamic World categories
    legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 300px; height: 150px;
                    border:2px solid grey; background-color:white; opacity: 0.7; z-index:9999; font-size:14px;
                    padding: 10px;
                    ">
        <b>Legend</b><br><br>
        <div style="display: flex; flex-wrap: wrap; width: 270px;">
            <div style="width: 33%; text-align: left;">
                <i style="background: #419bdf; width: 20px; height: 20px; display: inline-block;"></i> Water<br>
            </div>
            <div style="width: 33%; text-align: center;">
                <i style="background: #397d49; width: 20px; height: 20px; display: inline-block;"></i> Trees<br>
            </div>
            <div style="width: 33%; text-align: center;">
                <i style="background: #88b053; width: 20px; height: 20px; display: inline-block;"></i> Grass<br>
            </div>
            <div style="width: 33%; text-align: left;">
                <i style="background: #7a87c6; width: 20px; height: 20px; display: inline-block;"></i> Flooded<br>
            </div>
            <div style="width: 33%; text-align: center;">
                <i style="background: #e49635; width: 20px; height: 20px; display: inline-block;"></i> Crops<br>
            </div>
            <div style="width: 34%; text-align: center;">
                <i style="background: #dfc35a; width: 20px; height: 20px; display: inline-block;"></i> Shrubs<br>
            </div>
            <div style="width: 33%; text-align: left;">
                <i style="background: #c4281b; width: 20px; height: 20px; display: inline-block;"></i> Built<br>
            </div>
            <div style="width: 31%; text-align: center;">
                <i style="background: #a59b8f; width: 20px; height: 20px; display: inline-block;"></i> Bare<br>
            </div>
            <div style="width: 33%; text-align: center;">
                <i style="background: #b39fe1; width: 20px; height: 20px; display: inline-block;"></i> Snow<br>
            </div>
        </div>
        </div>
    '''
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def add_forest_change_legend(map_objfc):
    legend_html = '''
    <div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 250px;
    height: 100px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    opacity: 0.8;
    ">
    <b>Legend</b><br><br>
    <div style="background-color:yellow;width:20px;height:20px;float:left;margin-right:5px;"></div> Tree Loss Early<br>
    <div style="background-color:red;width:20px;height:20px;float:left;margin-right:5px;"></div> Tree Loss Recent<br>
    </div>
    '''
    map_objfc.get_root().html.add_child(folium.Element(legend_html))

# Function to get and clip Sentinel-2 imagery
def get_and_clip_sentinel_image(start_date, end_date, bbox_geom):
    try:
        region = ee.Geometry.Polygon(bbox_geom.bounds.tolist())

        sentinel_collection = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                               .filterDate(start_date, end_date)
                               .filterBounds(region)
                               .select('B4', 'B3', 'B2', 'B8')  # Added B8 for classification
                               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

        sentinel_image = sentinel_collection.median()
        clipped_image = sentinel_image.clip(region)  # Clip the image to the bounding box

        return clipped_image, region
    except Exception as e:
        st.error(f"Error fetching and clipping Sentinel-2 imagery: {e}")
        return None, None

def generate_eudr_report(start_date, end_date, croplands_within_county_and_type, forested_change_area_hectares=None):
    pdf = FPDF()

    # Add a page and set title
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Coffee Farmer EUDR Compliance Report', ln=True, align='C')


    pdf.cell(200, 10, f'Analysis: {start_date} to {end_date}', ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.multi_cell(0, 10,
    """This report summarizes the compliance of coffee farmers' in relation to the European Union Deforestation Regulations (EUDR).
Ensuring compliance with the European Union Deforestation Regulations (EUDR) is crucial for maintaining access to the EU market for agricultural products, particularly coffee.
Effective from December 31, 2024, these regulations require the collection and verification of geographic coordinates to confirm that commodities are not sourced from deforested areas. """)

    #displaying farms information; ID & Acreage
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, 'Farm Information:', ln=True, align='L')
    pdf.set_font('Arial','', 12)
    unique_farms = croplands_within_county_and_type['Bname'].unique()
    for farm_id in unique_farms:
        pdf.cell(200, 10, f'Farm ID: {farm_id}', ln=True, align='L')

    cropland_area_m2 = croplands_within_county_and_type.to_crs('EPSG:32737').area.sum()
    cropland_area_ha = cropland_area_m2 / 10000
    pdf.cell(200, 10, f'Total Area of the Farm: {cropland_area_ha:,.2f} Ha', ln=True, align='L')

    # Compliance status section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, 'Compliance status:', ln=True, align='L')
    pdf.set_font('Arial','', 12)
    pdf.ln(5)
    percentage_of_forest_loss = (forested_change_area_hectares / cropland_area_ha) * 100
    if forested_change_area_hectares and forested_change_area_hectares > 0:
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 10,
        f"""Non Compliant
- Tree Cover loss has been detected within the coffee farm area.
- Total Tree Loss is: {forested_change_area_hectares:,.2f} Ha i.e approximately {percentage_of_forest_loss:,.2f} % of the total farm area.
- This farm does not meet the EUDR Compliance Requirement.
- Further investigation & remediation required.""")

    else:
        pdf.set_text_color(0, 128, 0)
        pdf.multi_cell(0, 10,
        """Compliant
- No Tree Cover loss detected within the coffee farm area.
- This farm meets the EUDR Compliance Requirement.
- Continued monitoring recommended.""")


    # Create a buffer and return the PDF file
    pdf_buffer = io.BytesIO()
    pdf_data = pdf.output(dest='S').encode('latin1')
    pdf_buffer.write(pdf_data)
    pdf_buffer.seek(0)

    return pdf_buffer

def main():
    st.title("EUDR Coffee Farmers' Compliance Monitor")
        # Add geolocation and route drawing functionality
    location = streamlit_geolocation()

      # Load data for counties and croplands
    county = load_kenyan_counties()
    croplands = load_croplands()

    # Sidebar: Select the county
    OriginCounty = st.sidebar.selectbox("Select AOI", county['NAME'].unique())
    filtered_county = filter_county_by_name(county, OriginCounty)

    # Get geometry of selected county
    county_geometry = filtered_county.geometry.iloc[0]
    croplands_within_county = croplands[croplands.intersects(county_geometry)]
    available_crops = croplands_within_county['Bname'].unique()

    # Sidebar: Select AOI from available crops
    CountyCropland = st.sidebar.selectbox("Select Coffee Farm ID", available_crops)
    croplands_within_county_and_type = filter_cropland_by_county_and_type(croplands, filtered_county, CountyCropland)

    if location:
            user_lat, user_lon = location['latitude'], location['longitude']
            # Get the centroid of the selected coffee farm
            farm_centroid = croplands_within_county_and_type.geometry.centroid.iloc[0]
            farm_lat, farm_lon = farm_centroid.y, farm_centroid.x

    st.sidebar.write(f"User location: {user_lat:.3f}, {user_lon:.3f}")
    st.sidebar.write(f"Farm location: {farm_lat:.3f}, {farm_lon:.3f}")
    cropland_area_m2 = croplands_within_county_and_type.to_crs('EPSG:32737').area.sum()
    cropland_area_ha = cropland_area_m2 / 10000
    st.sidebar.write(f"Total Coffee Farm Area: {cropland_area_ha:,.3f} Ha")

    # Initialize variables to store clipped datasets
    ndvi_clipped = None
    sentinel_clipped = None
    forest_clipped = None
    lulc_clipped = None
    forest_change_clipped = None
    region = None


    # Sidebar: Defining the start and end date
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())

    if start_date >= end_date:
        st.error("End Date must be after Start Date.")
        return

    # Sidebar: Dataset selection buttons
    #First Column of buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        calculate_rgb_sentinel_button = st.button("Sentinel-2 MSI")
    with col2:
        calculate_ndvi_button = st.button("Calculate NDVI")

    # Calculate Sentinel-2 MSI
    if calculate_rgb_sentinel_button:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        sentinel_collection = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                               .filterDate(start_date_str, end_date_str)
                               .select('B4', 'B3', 'B2')
                               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        sentinel_image = sentinel_collection.median()
        sentinel = calculate_RGB(sentinel_image)
        croplands_ee = geodf_to_ee_featurecollection(croplands_within_county_and_type)
        sentinel_clipped = sentinel.clip(croplands_ee)
        st.success("Sentinel imagery clipped successfully!")

    # Calculate NDVI
    if calculate_ndvi_button:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        sentinel_collection = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                               .filterDate(start_date_str, end_date_str)
                               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        ndvi_collection = sentinel_collection.map(calculate_ndvi)
        ndvi_image = ndvi_collection.median()
        ndvi = calculate_ndvi(ndvi_image)
        croplands_ee = geodf_to_ee_featurecollection(croplands_within_county_and_type)
        ndvi_clipped = ndvi.clip(croplands_ee)
        st.success("NDVI calculated successfully!")


    # Sidebar: Additional dataset buttons
    #2nd Row of buttons
    col3, col4 = st.sidebar.columns(2)
    with col3:
        generate_lulc = st.button("Generate LULC")
    with col4:
        generate_forest = st.button("Tree Cover Area")

    if generate_lulc:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        lulc_image = generate_lulc_data(start_date_str, end_date_str)
        croplands_ee = geodf_to_ee_featurecollection(croplands_within_county_and_type)
        lulc_clipped = visualize_lulc(lulc_image.clip(croplands_ee))
        st.success("LULC generated successfully!")

    if generate_forest:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        forest_image = generate_forest_data(start_date_str, end_date_str)
        forest = generate_forestdata(forest_image)
        croplands_ee = geodf_to_ee_featurecollection(croplands_within_county_and_type)
        forest_clipped = forest.clip(croplands_ee)
        first_band = forest_clipped.bandNames().get(0)
        area_calculation = forest_clipped.multiply(ee.Image.pixelArea()) # Convert to km²
        area_dict = area_calculation.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=croplands_ee.geometry(),
            scale=30,
            maxPixels=1e9
        ).getInfo()

        # Get the area using the first band name
        forested_area_km2 = float(area_dict[first_band.getInfo()])
        forested_area_hectares = forested_area_km2 / 10000


        st.sidebar.info(f"Total Tree Cover area is: {forested_area_hectares:.2f} Ha")
        st.success("Tree Cover generated successfully!")
    sentinel_image, croplands_ee = None, None

     # Datasets selection part 3
    #3rd Row of buttons
    col5, col6 = st.sidebar.columns(2)
    with col5:
        generate_change_image = st.button("Tree Cover Change")

    if generate_change_image:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        forest_change_image = generate_forest_change()
        forest_change = generate_forestchange(forest_change_image)
        croplands_ee = geodf_to_ee_featurecollection(croplands_within_county_and_type)
        forest_change_clipped = forest_change.clip(croplands_ee)
        loss_mask = forest_change_clipped.select('lossyear').gt(0)
        area_calculation = loss_mask.multiply(ee.Image.pixelArea()) # Convert to km²
        area_dict = area_calculation.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=croplands_ee.geometry(),
            scale=30.92,
            maxPixels=1e9
        ).get('lossyear').getInfo()

        # Get the area using the first band name
        forested_change_area_hectares = area_dict / 10000
        st.sidebar.info(f"Total Tree Cover Loss area is: {forested_change_area_hectares:.2f} Ha")
        st.success("Tree Cover change generated successfully!")

        st.session_state.forest_change_area = forested_change_area_hectares
        st.session_state.start_date = start_date_str
        st.session_state.end_date = end_date_str


    with col6:
        geolocate_coffee = st.button("Geolocate Coffee Farm")

    if geolocate_coffee:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        sentinel_collection = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                               .filterDate(start_date_str, end_date_str)
                               .select('B4', 'B3', 'B2')
                               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        sentinel_image = sentinel_collection.median()
        sentinel = calculate_RGB(sentinel_image)
        # Ensure croplands_ee is valid and clip the image to the exact shape
        try:
            croplands_ee = geodf_to_ee_featurecollection(croplands_within_county_and_type)
            sentinel_clipped = sentinel.clipToCollection(croplands_ee)  # Clip precisely to the cropland shape

            st.success("Sentinel imagery clipped successfully!")
        except Exception as e:
            st.error(f"Error processing croplands or clipping: {e}")
            return

        # Define the output path to the Downloads folder
        downloads_folder = os.path.expanduser('~/Downloads')  # This works on most systems
        file_name = "Coffee_farm_geolocation.tif"
        output_path = os.path.join(downloads_folder, file_name)

        # Export the image to GeoTIFF format
        try:
            geometry = croplands_ee.geometry()

            geemap.ee_export_image(
                sentinel_clipped,
                filename=output_path,
                scale=10,
                region=geometry,  # Ensure it exports only the area within croplands geometry
                file_per_band=False
            )

            # Allow users to download the file manually through Streamlit
            with open(output_path, 'rb') as f:
                st.sidebar.download_button(label="Fetch Coffee Farm Imagery", data=f, file_name=file_name, mime="image/tiff")

        except Exception as e:
            st.error(f"Error exporting image: {e}")


    #4th Row of buttons
    col7, col8 = st.sidebar.columns(2)

    with col7:
        segment_coffee_areas = st.button("Fetch Coffee Farm Polygon")

    if segment_coffee_areas:
        # Convert the GeoDataFrame to GeoJSON
        geojson_data = croplands_within_county_and_type.to_json()

        # Create a download button for the GeoJSON file
        st.sidebar.download_button(
            label="Download Polygon",
            data=geojson_data,
            file_name="Coffee_Farm.geojson",
            mime="application/json"
        )

        st.success("Coffee Farm fetched successfully! Click the download button to save the polygon.")

    with col8:
        download_report = st.button("EUDR Report")

    if download_report:
        forest_change_area = st.session_state.get('forest_change_area', None)
        start_date_str = st.session_state.get('start_date', start_date.strftime('%Y-%m-%d'))
        end_date_str = st.session_state.get('end_date', end_date.strftime('%Y-%m-%d'))

        pdf_buffer = generate_eudr_report(start_date_str, end_date_str, croplands_within_county_and_type,  forest_change_area)

        st.sidebar.download_button(
            label="Download EUDR Compliance Report",
            data= pdf_buffer,
            file_name=f"EUDR_Compliance_Report_{start_date_str}_{end_date_str}.pdf",
            mime="application/pdf"
        )

    # Create geemap map
    geemap_map = create_geemap(OriginCounty, county, croplands_within_county_and_type, forest_clipped, ndvi_clipped, forest_change_clipped,
                               sentinel_clipped)

    # Add LULC layer if available
    if lulc_clipped:
        geemap_map.addLayer(lulc_clipped, {}, 'LULC')
        add_lulc_legend(geemap_map)

    if geemap_map is not None:
        geemap_map.to_streamlit(height=600, width=1230)

    # Copyright
    st.markdown("&copy; Peter Thuku 2024")


if __name__ == "__main__":
    main()



