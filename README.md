# 4DLangevin Fission Fragment Data

Physical processes in nuclear fission are of research interest due to the complexity of the large amplitude collective motion in the compound nucleus. We have developed a four-dimensional (4D) Langevin model with the potential energy calculated using the deformed two-center Woods-Saxon (TCWS) potential and the Nilsson-type potential, including microscopic energy corrections following the Strutinsky method and BCS pairing [C. Ishizuka et al. Phys. Rev. C 96 (2017)](http://dx.doi.org/10.1080/00223131.2018.1467288). This repository contains datasets of yields and total kinetic energy (TKE) of fission fragment pairs from even-even compound nuclei from U up to Z=120 at 10 MeV excitation energy, calculated using the 4D Langevin model. We hope that this dataset can provide opportunities for researchers to benchmark fission fragment decay simulations, train machine learning models, or develop correlations to gain new insights into the physical processes in nuclear fission.

## About Data collection methodology

The fission event data from the 4D Langevin model is followed by data processing to calculate the mass yield and Gaussian distribution of TKE using a simple Python code.

### Description of the data

Here you can descibe how the data is organized in this whole dataset. How the data is stored in all the files. You also have to brief about the naming convention of the files in different directories. 

Dataset names follow the convention: 
    * `CompoundNuclideElementMass_ExcitationEnergy_TKE.dat` - Y(A, TKE, dTKE) data by Gaussian fitting of TKE distribution at each fission fragment mass
    * `CompoundNuclideElementMass_ExcitationEnergy_fullTKE.dat` - full A(Y, TKE) data purely from 4d Langevin model. TKE is binned in 1 MeV energy bin.

```
4DLangevin
  ├── README.md
  ├── files/
  │   ├── Fl/
  │   │    ├── Fl276_E10.0MeV_TKE.dat
  │   │    ├── Fl276_E10.0MeV_fullTKE.dat
  │   │    └── ...
  │   └── ...
  └── plots/
      ├── Fl/
      │    ├── 
      │    ├── Fl276_E10.0MeV_a_tke.png
      │    ├── Fl276_E10.0MeV_tke.png
      │    ├── Fl276_E10.0MeV_ya.png
      │    └── ...
      └── ...
```

* Please note that files with charge and fragment excitation energies, i.e. Y(Z, A, TKE, TXE, Eex_h, dEex_h, , Eex_l, dEex_l), will be available in the nuclear reaction code TALYS package.


### File formats
Both data files only contain data on heavy fragment masses, as shown in the following examples.

`CompoundNuclideElementMass_ExcitationEnergy_TKE.dat` 
```
#  A       Yield            Mean TKE         dTKE
  138      6.71848E-02      2.68184E+02      1.49124E+01
  139      7.56412E-02      2.66829E+02      1.66315E+01
```

`CompoundNuclideElementMass_ExcitationEnergy_fullTKE.dat`
```
#  A       TKE[MeV]          Yield
  138      2.03000E+02      0.00000E+00
  138      2.04000E+02      0.00000E+00
  138      2.05000E+02      9.34754E-05
  138      2.06000E+02      0.00000E+00
  138      2.07000E+02      9.34754E-05
  138      2.08000E+02      0.00000E+00
  138      2.09000E+02      0.00000E+00
  138      2.10000E+02      0.00000E+00
  138      2.11000E+02      0.00000E+00
  138      2.12000E+02      9.34754E-05
  138      2.13000E+02      9.34754E-05
  138      2.14000E+02      9.34754E-05
   :
  139      2.03000E+02      4.67377E-05
  139      2.04000E+02      4.67377E-05
  139      2.05000E+02      0.00000E+00
  139      2.06000E+02      0.00000E+00
  139      2.07000E+02      0.00000E+00
  139      2.08000E+02      4.67377E-05
  139      2.09000E+02      0.00000E+00
  139      2.10000E+02      9.34754E-05
  139      2.11000E+02      1.40213E-04
  139      2.12000E+02      1.40213E-04
  139      2.13000E+02      4.67377E-05
  139      2.14000E+02      0.00000E+00
```

## Download
You can download the repository from the terminal using the following command:
```
git clone https://github.com/chikako-ishizuka/4DLangevin.git
```


## Usage
Examples for downloading and loading into Pandas Dataframe to plot the data:

```python
import pandas as pd
data_url = 'https://raw.githubusercontent.com/chikako-ishizuka/4DLangevin/main/files/Fl/Fl276_E10.0MeV_TKE.dat'
df = pd.read_csv(
                    data_url, 
                    comment='#', 
                    names = ["A_H", "Yield",  "TKE",  "dTKE"], 
                    header=None, 
                    delim_whitespace=True
                   )
print(df)

plt.ylabel('Yield')
plt.xlabel('FF Mass Number')
df.plot("A_H", "Yield", color="b")
plt.show()
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

data_url = 'https://raw.githubusercontent.com/chikako-ishizuka/4DLangevin/main/files/Fl/Fl276_E10.0MeV_fullTKE.dat'
df = pd.read_csv(
                    data_url, 
                    comment='#', 
                    names = ["A_H", "TKE", "Yield"], 
                    header=None, 
                    delim_whitespace=True
                   )
print(df)

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

ax.set_xlabel('FF Mass Number')
ax.set_ylabel('TKE')
ax.set_zlabel("Yield")

x = df['A_H']
y = df['TKE']
z = df['Yield']
xi = np.linspace(x.min(), x.max(), len(x.unique()))
yi = np.linspace(y.min(), y.max(), len(y.unique()))
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')
surf = ax.plot_surface(xi, yi, zi, cmap = plt.cm.cividis, shade=False)
plt.show()
```

## Online Repository link

* [DataRepository](https://github.com/chikako-ishizuka/4DLangevin) - Link to the data repository.


## Authors

* **Chikako Ishizuka, Mark D. Usang, Fedir A. Ivanyuk, Joachim A. Maruhn, Katsuhisa Nishio, and Satoshi Chiba** - *Initial work* - [Phys. Rev. C 96 (2017)](http://dx.doi.org/10.1080/00223131.2018.1467288.) 


## License and Disclaimer

This project is licensed under the CC BY-SA License - see the [LICENSE.md](LICENSE.md) file for details

* Data Citation: Data sources should be acknowledged by a citation. 
    ```
    Ishizuka, C. ; Usang, M. D. ; Ivanyuk, F. A. ; Maruhn, J. A. ; Nishio, K. ; Chiba, S. : Four-dimensional Langevin approach to low-energy nuclear fission of 236U. In: Phys. Rev. C 96 (2017), Dec, 064616.
    ```
* Reporting Issues: If you find any issues of dataset, please contact  
* Changes: The developers of this repository reserve the right to modify or update datasets at any time without prior notice.

## Changelog
August 2023

