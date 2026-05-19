import os
import dicom2nifti
import numpy as np
import pydicom
from datetime import datetime, timedelta
import pandas as pd
import shutil
import nibabel as nib

def rename_folders(root_folder, mapping_file):
    """Rename folders based on mapping CSV file, including PET-CT/PET subfolder"""
    df = pd.read_csv(mapping_file)
    # Convert AnonymizedID to string to match folder names
    mapping = dict(zip(df['AnonymizedID'].astype(str), df['PID']))
    
    for old_name in os.listdir(root_folder):
        if old_name in mapping:
            old_path = os.path.join(root_folder, old_name)
            new_path = os.path.join(root_folder, mapping[old_name])
            if os.path.exists(old_path) and not os.path.exists(new_path):
                shutil.move(old_path, new_path)
                print(f"Renamed {old_name} to {mapping[old_name]}")
                # Also rename PET-CT/PET subfolder if it exists
                pet_subfolder = os.path.join(new_path, 'PET-CT', 'PET', old_name)
                new_pet_subfolder = os.path.join(new_path, 'PET-CT', 'PET', mapping[old_name])
                if os.path.exists(pet_subfolder) and not os.path.exists(new_pet_subfolder):
                    shutil.move(pet_subfolder, new_pet_subfolder)
                    print(f"Renamed PET subfolder {pet_subfolder} to {new_pet_subfolder}")

def load_first_dicom(folder_path):
    """Load the first PT DICOM file for metadata extraction"""
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            try:
                ds = pydicom.dcmread(filepath)
                if ds.Modality == "PT":
                    return ds
            except Exception as e:
                print(f"Skipped {filename}: {e}")
    raise ValueError("No valid PT DICOM files found.")

def get_acquisition_datetime(ds):
    date = ds.StudyDate
    time = ds.AcquisitionTime.split('.')[0]
    return datetime.strptime(date + time, "%Y%m%d%H%M%S")

def get_injection_datetime(info):
    date = info.RadiopharmaceuticalStartTime
    if '.' in date:
        date = date.split('.')[0]
    return datetime.strptime(date, "%H%M%S")

def extract_suv_metadata(ds):
    """Extract all metadata needed for SUV conversion"""
    acquisition_datetime = get_acquisition_datetime(ds)
    info = ds.RadiopharmaceuticalInformationSequence[0]
    
    # Check pixel value units
    units = getattr(ds, 'Units', 'UNKNOWN')
    if units != 'BQML':
        raise ValueError(f"Pixel units are '{units}', expected 'BQML'. Additional calibration may be needed.")
    
    # Check if decay correction has already been applied
    corrected_image = getattr(ds, 'CorrectedImage', [])
    if isinstance(corrected_image, str):
        corrected_image = [corrected_image]
    is_decay_corrected = 'DECY' in corrected_image
    
    injected_dose = float(info.RadionuclideTotalDose)  # in Bq
    half_life = float(info.RadionuclideHalfLife)  # in seconds
    patient_weight_kg = float(ds.PatientWeight)
    patient_weight_g = patient_weight_kg * 1000
    
    # Time difference calculation and decay correction
    if is_decay_corrected:
        decay_corrected_dose = injected_dose
    else:
        decay_constant = np.log(2) / half_life
        inj_time = get_injection_datetime(info)
        inj_datetime = datetime.combine(acquisition_datetime.date(), inj_time.time())
        if inj_datetime > acquisition_datetime:
            inj_datetime -= timedelta(days=1)
        
        delta_seconds = (acquisition_datetime - inj_datetime).total_seconds()
        decay_corrected_dose = injected_dose * np.exp(-decay_constant * delta_seconds)
    
    return {
        'patient_weight_g': patient_weight_g,
        'decay_corrected_dose': decay_corrected_dose,
        'acquisition_datetime': acquisition_datetime,
        'is_decay_corrected': is_decay_corrected,
        'units': units
    }

def convert_bq_to_suv(nifti_path, suv_metadata, output_path):
    """Load NIfTI, convert from Bq to SUV, and save"""
    # Load the NIfTI file
    img = nib.load(nifti_path)
    bq_data = img.get_fdata()
    
    # Apply SUV conversion: SUV = (Bq/ml * patient_weight_g) / decay_corrected_dose
    suv_data = (bq_data * suv_metadata['patient_weight_g']) / suv_metadata['decay_corrected_dose']
    
    # Create new NIfTI with SUV data
    suv_img = nib.Nifti1Image(suv_data, img.affine, img.header)
    nib.save(suv_img, output_path)
    
    return suv_data.shape

def convert_patient_data(root_folder, target_folder, mapping_file=None, single_patient=None):
    """
    Converts PET DICOM to NIfTI with SUV conversion. If mapping_file is provided, renaming is performed first.
    If single_patient is provided, only that patient is processed.
    """
    if mapping_file:
        rename_folders(root_folder, mapping_file)
    
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    for patient_folder in os.listdir(root_folder):
        if single_patient and patient_folder != single_patient:
            continue
        patient_path = os.path.join(root_folder, patient_folder)
        if not os.path.isdir(patient_path):
            continue
        
        # Convert PET
        pet_ct_folder = os.path.join(patient_path, 'PET-CT')
        print(f"Processing patient: {patient_folder}")
        if os.path.isdir(pet_ct_folder):
            pet_folder = os.path.join(pet_ct_folder, 'PET')
            if os.path.isdir(pet_folder):
                try:
                    # Step 1: Convert DICOM to NIfTI (Bq values)
                    temp_nifti = os.path.join(target_folder, f'{patient_folder}__PT_temp.nii.gz')
                    dicom2nifti.dicom_series_to_nifti(pet_folder, temp_nifti)
                    
                    # Step 2: Extract SUV metadata from DICOM
                    ds = load_first_dicom(pet_folder)
                    suv_metadata = extract_suv_metadata(ds)
                    
                    # Step 3: Load NIfTI, convert Bq→SUV, save final result
                    final_output = os.path.join(target_folder, f'{patient_folder}__PT.nii.gz')
                    suv_shape = convert_bq_to_suv(temp_nifti, suv_metadata, final_output)
                    
                    # Clean up temp file
                    os.remove(temp_nifti)
                    
                    decay_status = "pre-corrected" if suv_metadata['is_decay_corrected'] else "corrected"
                    print(f"✓ Successfully converted {patient_folder} (shape: {suv_shape}, decay: {decay_status})")
                    
                except Exception as e:
                    print(f"✗ Error processing {patient_folder}: {e}")
            else:
                print(f"✗ PET folder not found for {patient_folder}")
        else:
            print(f"✗ PET-CT folder not found for {patient_folder}")

if __name__ == "__main__":
    root_folder = "/Users/numansaeed/Downloads/final_data/anonymized_data/new_patients/anonymized_DICOM"
    target_folder = "/Users/numansaeed/Downloads/final_data/anonymized_data/new_patients/SUV_NIFTI_CORRECTED"
    mapping_file = "MDA_new_case_mapping.csv"

    # Always rename first
    rename_folders(root_folder, mapping_file)

    # Run for all patients
    print("Converting all patients with corrected SUV calculation...")
    convert_patient_data(root_folder, target_folder, mapping_file=None)