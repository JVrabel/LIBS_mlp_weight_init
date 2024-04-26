import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

def load_h5_data(path):
    """
    Function for loading h5 format of the ChemCam extended calibration dataset.

    Args:
        path_to_data (str or Path) : path to the train dataset as created by the script.

    Returns:
        pd.DataFrame : X
        pd.Series : y
        entry to the file it originated from. 
    """
    loaded_data={
        'metadata':pd.read_hdf(path,key='metadata'),
        'spectra':pd.read_hdf(path,key='spectra')
    }

    with h5py.File(
      path,
      'r'
    ) as file:
        loaded_data['wvl'] = file['wvl'][()]


    spectra = loaded_data['spectra']
    spectra = spectra.iloc[:,1:]
    metadata = loaded_data['metadata']
    wavelengths = loaded_data['wvl']
    wavelengths = wavelengths[1:]

    return pd.DataFrame(np.array(spectra), columns=wavelengths), metadata 




def load_contest_train_dataset(path_to_data, spectra_per_sample: int=50):
  """
  Function for loading the contest train dataset.

  Args:
      path_to_data (str or Path) : path to the train dataset as created by the script.
      spectra_per_sample (int) : how many spectra will be taken from each sample.

  Returns:
      pd.DataFrame : X
      pd.Series : y
      pd.Series : list of sample labels for each entry. Can be used to connect each 
      entry to the file it originated from. 
  """
  if isinstance(path_to_data, str):
    path_to_data = Path(path_to_data)
  with h5py.File(path_to_data, 'r') as train_file:
    # Store wavelengths (calibration)
    wavelengths = pd.Series(train_file['Wavelengths']['1'])
    wavelengths = wavelengths.round(2).drop(index=[40000, 40001])

    # Store class labels
    labels = pd.Series(train_file['Class']['1']).astype(int)

    # Store spectra
    samples_per_class = labels.value_counts(sort=False) // 500
    spectra = np.empty(shape=(0, 40000))
    samples = []
    classes = []

    lower_bound = 1
    for i_class in tqdm(samples_per_class.keys()):
      for i_sample in range(lower_bound, lower_bound + samples_per_class[i_class]):
        sample = train_file["Spectra"][f"{i_sample:03d}"]
        sample = np.transpose(sample[:40000, :spectra_per_sample])
        spectra = np.concatenate([spectra, sample])
        samples.extend(np.repeat(i_sample, spectra_per_sample))
        classes.extend(np.repeat(i_class, spectra_per_sample))
      lower_bound += samples_per_class[i_class]

  samples = pd.Series(samples)
  classes = pd.Series(classes)
  return pd.DataFrame(spectra, columns=wavelengths), classes, samples




def load_contest_test_dataset(path_to_data):
  testFile = h5py.File(path_to_data,'r')
  for chunk in list(testFile["UNKNOWN"].keys()):
      tempData = np.array(testFile["UNKNOWN"][chunk])
      if "testData" not in locals():
          testData = tempData.transpose()
      else:
          testData = np.append(testData, tempData.transpose(), axis = 0)
  testData = testData[:,:40000]
  return pd.DataFrame(testData)


