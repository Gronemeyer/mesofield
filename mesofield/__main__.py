import os
import logging

import click

# Disable pymmcore-plus logger
package_logger = logging.getLogger('pymmcore-plus')
package_logger.setLevel(logging.CRITICAL)

# Disable debugger warning about the use of frozen modules
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Disable ipykernel logger
logging.getLogger("ipykernel.inprocess.ipkernel").setLevel(logging.WARNING)

def launch_mesofield(params):
    from PyQt6.QtWidgets import QApplication
    from mesofield.gui.maingui import MainWindow
    from mesofield.config import ExperimentConfig

    """Launch the mesofield acquisition interface."""
    print('Launching mesofield acquisition interface...')
    app = QApplication([])
    config = ExperimentConfig(params)
    config.hardware._configure_engines(config)
    mesofield = MainWindow(config)
    mesofield.show()
    app.exec()


def convert_pupil_videos_to_mp4(parent_dir):
    """Convert the pupil videos to mp4 format."""
    from mesofield.data.batch import tiff_to_mp4
        
    tiff_to_mp4(
        parent_directory=parent_dir,
        fps=30,
        output_format="mp4",
        use_color=False
    )

def get_experimental_summary(experiment_dir):
    import mesofield.data.load as load
    
    datadict =  load.file_hierarchy(experiment_dir)
    load.experiment_progress_summary(datadict)


def get_file_hierarchy_object(experiment_dir):
    import mesofield.data.load as load
    
    return load.file_hierarchy(experiment_dir)


def get_mean_trace(experiment_dir, subject_id):
    import pandas as pd
    import mesofield.data.load as load
    import mesofield.data.batch as batch
    
    datadict =  load.file_hierarchy(experiment_dir)
    session_paths = []
    for key in sorted(datadict[subject_id].keys()):
        if key.isdigit():
            session_paths.append(datadict[subject_id][key]['meso']['tiff'])
    
    results = batch.mean_trace_from_tiff(session_paths)
    for path, trace in results.items():
        print(f"{path}: {trace[:10]}") 
        
    outdir = os.path.join(experiment_dir, "processed", subject_id)
    os.makedirs(outdir, exist_ok=True)

    for path, trace in results.items():
        df = pd.DataFrame({"Slice": range(len(trace)), "Mean": trace})
        base_name = os.path.splitext(os.path.basename(path))[0]
        filename = f"{base_name}_meso-mean-trace.csv"
        df.to_csv(os.path.join(outdir, filename), index=False)
    
    #print(data)



def plot_session_data(experiment_dir, subject_id, session_id, save):
    """Plot the session data."""
    import pandas as pd
    import mesofield.data.load as load
    import mesofield.data.transform as transform
    import mesofield.data.plot as plot
    
    datadict =  load.file_hierarchy(experiment_dir)
    meso_file = datadict[subject_id][session_id]['processed']['meso_trace']
    pupil_dlc_file = datadict[subject_id][session_id]['processed']['dlc_pupil']
    encoder_file = datadict[subject_id][session_id]['encoder']
    
    print(f"Meso file: {meso_file}")
    print(f"Pupil file: {pupil_dlc_file}")
    print(f"Encoder data: {encoder_file}")
    
    meso_df = pd.read_csv(meso_file)
    encoder_df = pd.read_csv(encoder_file)

    pupil_df = transform.process_deeplabcut_pupil_data(pickle_path=pupil_dlc_file,
                                               show_plot=False,
                                               confidence_threshold=0.7)
    # APPLY OPTIONAL FILTERS --------------------------------------------------
    encoder_df = transform.apply_filters(encoder_df, 
                                         speed_col='Speed', 
                                         threshold=0.001, 
                                         smoothing='ewm', 
                                         window_size=5)
    # ------------------------------------------------------------------------
    meso_metadata = load.camera_metadata(datadict[subject_id][session_id]['meso']['metadata'])
    pupil_metadata = load.camera_metadata(datadict[subject_id][session_id]['pupil']['metadata'])

    meso_df = meso_df.join(meso_metadata)
    meso_df = meso_df.join(encoder_df['Speed']).rename(columns={'Speed': 'encoder_speed'})
    pupil_df = pupil_df.join(pupil_metadata)

    plot = plot.plot_session(session_name=f"{subject_id} - {session_id}",
                             df_fluorescence=meso_df,
                             df_encoder=meso_df,
                             df_pupil=pupil_df,
                             fluorescence_x='Slice', 
                             fluorescence_y='Mean',
                             speed_col='encoder_speed',
                             locomotion_threshold=0.01,
                             downsample=10, 
                             x_limit=(0, len(meso_df)))  # adjust as desired, or None
    plot.show()
    if save:
        plot.save()


def test_psychopy():
    import sys
    from PyQt6.QtWidgets import QApplication
    import tests.test_psychopy as test_psychopy
    
    app = QApplication(sys.argv)
    gui = test_psychopy.DillPsychopyGui()
    gui.show()
    sys.exit(app.exec())
    
def test_mda():
    from tests.test_mda import test_mmcore_mda

    test_mmcore_mda()

'''
================================== Command Line Interface ======================================
Commands:
    launch: Launch the mesofield acquisition interface
        --params: Path to the config file
        
    batch_pupil: Convert the pupil videos to mp4 format
        --dir: Directory containing the BIDS formatted /data hierarchy
        
    plot_session: Plot the session data
        --dir: Path to experimental directory containing BIDS formatted /data hierarchy
        --sub: Subject ID (the name of the subject folder)
        --ses: Session ID (two digit number indicating the session)
        --save: Save the plot to the processing directory in the Experiment folder
        
    trace_meso: Get the mean trace of the mesoscopic data
        --dir: Path to experimental directory containing BIDS formatted /data hierarchy
        --sub: Subject ID (the name of the subject folder)
        
'''
@click.group()
def cli():
    """mesofields Command Line Interface"""

@cli.command()
@click.option('--params', default='hardware.yaml', help='Path to the config file')
def launch(params):
    launch_mesofield(params)

@cli.command()
@click.option('--dir', help='Path to experimental directory containing BIDS formatted /data hierarchy')
@click.option('--sub', help='Subject ID (the name of the subject folder)')
@click.option('--ses', help='Session ID (two digit number indicating the session)')
@click.option('--save', default=False, help='Save the plot to the processing directory in the Experiment folder')
def plot_session(dir, sub, ses, save):
    plot_session_data(dir, sub, ses, save)

@cli.command()
@click.option('--dir', help='Save the plot to the processing directory in the Experiment folder')
@click.option('--sub', help='Subject ID (the name of the subject folder)')
def trace_meso(dir, sub):
    get_mean_trace(dir, sub)

@cli.command()
@click.option('--dir', help='Directory containing the BIDS formatted /data hierarchy')
def batch_pupil(dir):
    convert_pupil_videos_to_mp4(dir)

@cli.command()
def psychopy():
    test_psychopy()

@cli.command()
def mda():
    test_mda()
    
@cli.command()
@click.option('--dir', help='Directory containing the BIDS formatted /data hierarchy')
def load_exp_data(dir):
    from mesofield.data.load import load_data
    data = load_data(dir)
    
@cli.command()
@click.option('--exp_dir', default="c:/path/to/your/experimental_directory",
                help="Path to the experimental data directory.")
@click.option('--db_file', default="experiment.db",
                help="SQLite database file to create.")
def create_db(exp_dir, db_file):

    from mesofield.data.base import create_connection, create_table, insert_subject, insert_session, insert_file
    from mesofield.data.load import file_hierarchy
    from datetime import datetime
    
    conn = create_connection(db_file)
    if conn:
        create_table(conn)
        # Build the hierarchical datadict from the experimental directory.
        datadict = file_hierarchy(exp_dir)

        # Iterate over subjects and sessions to insert data into the database.
        for subject, sessions in datadict.items():
            subject_id = insert_subject(conn, subject)
            for session, categories in sessions.items():
                session_id = insert_session(conn, subject_id, session)
                # Process each category – either a list, dict, or string.
                for category, item in categories.items():
                    if isinstance(item, list):
                        for file_record in item:
                            insert_file(
                                conn,
                                session_id,
                                file_record["Filename"],
                                file_record["Path"],
                                file_record["Size (GB)"],
                                file_record["Modified Date"]
                            )
                    elif isinstance(item, dict):
                        for subcat, file_path in item.items():
                            if os.path.exists(file_path):
                                filename = os.path.basename(file_path)
                                size = os.path.getsize(file_path) / (1024 ** 3)
                                mod_time = os.path.getmtime(file_path)
                                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                filename = os.path.basename(file_path)
                                size = None
                                mod_date = None
                            insert_file(
                                conn,
                                session_id,
                                filename,
                                file_path,
                                size,
                                mod_date
                            )
                    elif isinstance(item, str):
                        file_path = item
                        if os.path.exists(file_path):
                            filename = os.path.basename(file_path)
                            size = os.path.getsize(file_path) / (1024 ** 3)
                            mod_time = os.path.getmtime(file_path)
                            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            filename = os.path.basename(file_path)
                            size = None
                            mod_date = None
                        insert_file(
                            conn,
                            session_id,
                            filename,
                            file_path,
                            size,
                            mod_date
                        )

        conn.commit()
        print("Database generated successfully.")
    else:
        print("Error! Cannot create the database connection.")

@cli.command()
@click.option('--db_file', default="experiment.db",
                help="SQLite database file to query.")
@click.option('--subject', default="STREHAB01",
                help="Subject name to query.")
@click.option('--session', default="01",
                help="Session name to query.")
def query_db(db_file, subject, session):
    from mesofield.data.base import create_connection
    from mesofield.data.base import query_files

    conn = create_connection(db_file)
    if conn:
        records = query_files(conn, subject)
        if records:
            for rec in records:
                print(rec)
        else:
            print("No records found.")
    else:
        print("Error! Cannot create the database connection.")

if __name__ == "__main__":
    cli()
