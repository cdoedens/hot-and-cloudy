import subprocess
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

script_to_run = 'CONCURRENT-FUTURES-SOLAR'

regions = [
    # 'GCCSA_1GSYD',
    # 'GCCSA_2GMEL',
    # 'GCCSA_3GBRI',
    'GCCSA_4GADE',
    # 'GCCSA_6GHOB',
    # 'GCCSA_8ACTE',
]
tilt = 'fixed'
for region in regions:
    # First full month of data (Aug '15) to latest (Jan '25)
    first = '1-8-2015'
    num_batches = 12 * 9 + 6

    first_dt = datetime.strptime(first, "%d-%m-%Y")
    dates = []
    for x in range(num_batches):
        start_dt = first_dt + relativedelta(months =  x)
        start_date = start_dt.strftime("%d-%m-%Y")
        end_dt = start_dt + relativedelta(months = 1) - relativedelta(days=1)
        end_date = end_dt.strftime("%d-%m-%Y")
        dates.append((start_date, end_date))

    if region[1].upper() == 'ALL':
        ncpus = 13
    else:
        ncpus = 48
    for start, end in dates:
        
        # Generate a unique file name based on iteration
        joboutdir = '/home/548/cd3022/hot-and-cloudy/code/qsub/'
        job_script_filename = joboutdir + f'{script_to_run}___' + start + '_' + region + '.qsub'
        
        # Open the file for writing
        with open(job_script_filename, "w") as f3:
            f3.write('#!/bin/bash \n')
            f3.write('#PBS -l walltime=1:00:00 \n')
            f3.write('#PBS -l mem=192GB \n')
            f3.write(f'#PBS -l ncpus={ncpus} \n')
            f3.write('#PBS -l jobfs=10GB \n')
            f3.write('#PBS -l storage=gdata/xp65+gdata/gb02+gdata/rv74+gdata/ob53 \n')
            f3.write('#PBS -l other=hyperthread \n')
            f3.write('#PBS -q normal \n')
            f3.write('#PBS -P er8 \n')
            f3.write(f'#PBS -o /home/548/cd3022/hot-and-cloudy/logs/{script_to_run}_{region}_{start}.oe \n')
            f3.write('#PBS -j oe \n')
            f3.write('module use /g/data3/xp65/public/modules \n')
            f3.write('module load conda/analysis3 \n')
            f3.write('conda \n')
            f3.write(f'python3 /home/548/cd3022/hot-and-cloudy/code/{script_to_run}.py {start} {end} {region} {tilt}\n')


        # Submit the generated script to the job scheduler (PBS) using qsub
        try:
            # Run the qsub command and submit the script
            subprocess.run(['qsub', job_script_filename], check=True)
            print(f"Job script {job_script_filename} submitted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job script {job_script_filename}: {e}")