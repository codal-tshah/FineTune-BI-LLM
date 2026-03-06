import vanna as vn
from connections import get_vanna_instance, connect_database

# Initialize dynamic vanna and database connection
vn = get_vanna_instance()
connect_database(vn)