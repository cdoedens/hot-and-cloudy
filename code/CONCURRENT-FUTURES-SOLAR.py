import concurrent.futures
import os,sys
from datetime import datetime, timedelta
from pathlib import Path

modules_path = Path(__file__).resolve().parent.parent / 'modules'
sys.path.append(str(modules_path))

from compute_solar2 import solar_workflow
import logger

LOG = logger.get_logger(__name__)

if __name__ == '__main__':
    
    n_procs = os.cpu_count()
    worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=10)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    region = sys.argv[3]
    tilt = sys.argv[4]
    
    start_dt = datetime.strptime(start_date, "%d-%m-%Y")
    end_dt = datetime.strptime(end_date, "%d-%m-%Y")
    
    # Generate a list of dates
    date_range = [start_dt + timedelta(days=i) for i in range((end_dt - start_dt).days + 1)]
    
    futures = {}
    # Loop over the dates
    for date in date_range: 
        date_s = date.strftime('%Y/%m/%d')
        LOG.info(f'Start workflow with date {date_s}, region {region}, tilt {tilt}')
        future = worker_pool.submit(solar_workflow, date_s, region, tilt) 
        futures[future] = f"The job for {date}"