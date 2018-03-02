


# Ptychography Alignment Tools


## Description

This project provides a PyQtGraph-based GUI to assist users on the alignment of Ptychography scans. The tool has the following features:
* Load set of images (tiff files supported)
* Select pairs of images for alignment
* Import/Export probe positions (npy array)
* Image controls: levels, contrast, look up tables, zooming, translation
* Preview the global picture by combining all positions

## Requirements
* [Python 2.7](https://www.python.org/)
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5)
* [PyQtGraph](http://www.pyqtgraph.org/)
* [Tifffile](https://pypi.python.org/pypi/tifffile)
* [Matplotlib](https://matplotlib.org/)
* [Numpy](http://www.numpy.org/)

## Installation


Be sure to have installed all the requirements.
Open a terminal and clone the repository to a local directory:
```
mkdir workdir
cd workdir
git clone url

```
To start the GUI change to src directory and run with Python:
```
cd PtychoAlignTools/src/
python ptychoAlign.py
```

## Usage
Once opened, the GUI will initialize like shown below:


![alt text](https://github.com/ElettraSciComp/PtychoAlignTools/blob/master/pictures/ptychoAlign_GUI_A.png)
 

 **Load Probes**
* By clicking on "Load Probes" a file dialog will prompt allowing for opening up to 12 images of probe scans and load them as a 4x3 map of single reconstructed probes.


 ![alt text](https://github.com/ElettraSciComp/PtychoAlignTools/blob/master/pictures/ptychoAlign_GUI_B.png)

 
* The two rectangles, red and green, allows for selecting a pair of images to be aligned. The selection is made by clicking on the probes. Right click to select an "movable" image and left click to select and "anchored" (not movable) image.
* The "Pairwise Alignment View" window display the selected images by doing an operation (Multiplication, Division, Addition, Subtraction, GK) between them.

**Load and Save Alignment**
* By clicking on "Load Alignment" button a file dialog will prompt. It allows to load a ".npy" file (numpy binary file) containing positions of previous alignment. We provide two sample files at [probe-positions](https://github.com/ElettraSciComp/PtychoAlignTools/tree/master/sample-data/probe-positions).
* The "Save Alignment" button allows to save the current alignment (displayed in the table at the "Positions" window) as a ".npy" file.
* See [numpy documentation](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.io.html) for details about ".npy" extension.

**Refresh View**
* Restore original size and histogram levels of the image displayed in "Pairwise Alignment View" window.

**Preview Global Image**
* Based on the current positions, combine all the 12 probes and show a preview of the total alignment.
 
 ![alt text](https://github.com/ElettraSciComp/PtychoAlignTools/blob/master/pictures/ptychoAlign_GUI_C.png)

**Load Mask**
* This feature allows a user to pick a mask and apply it to the selected probes in order improve the refinement of the positions. 
* Loading the "filter_gaussian_sigma20.png" mask provided in [masks](https://github.com/ElettraSciComp/PtychoAlignTools/tree/master/sample-data/masks) and setting the operation dropdown menu to "GK" give us the following:
  * place a screenshot.
* If a mask is loaded it will also be used on the display of the Global Image.


## Image processing and measure calculation
In the [auto](https://github.com/ElettraSciComp/PtychoAlignTools/tree/master/src/auto) subfolder there are some scripts related to:

* Probe alignment based on edge-type features

 ![alt text](https://github.com/ElettraSciComp/PtychoAlignTools/blob/master/pictures/edge_align.png)

* Probe alignment based on local-type features

 ![alt text](https://github.com/ElettraSciComp/PtychoAlignTools/blob/master/pictures/localFeat.png)

* Plotting of many measure in the neighborhood of a given position

 ![alt text](https://github.com/ElettraSciComp/PtychoAlignTools/blob/master/pictures/measures.png)

## References

[1] Francesco Guzzi, George Kourousias, Fulvio Billè, Roberto Pugliese, Carlos Reis, Alessandra Gianoncelli, and Sergio Carrato, "*Refining scan positions in Ptychography by using multiple metrics and Machine Learning*", submitted to *[JINST](https://jinst.sissa.it/jinst/help/helpLoader.jsp?pgType=about) Proceedings of the 24th International congress on x-ray optics and microanalysis*, 24-29 September 2017, Trieste, Italy ([ICXOM24](https://jinst.sissa.it/jinst/common/JINST_proceedings8.jsp)).
DOI:

**Data and reconstructions as presented in:**

[2] George Kourousias, Benedetto Bozzini, Alessandra Gianoncelli, Michael W. M. Jones, Mark Junker, Grant van Riessen, and Maya Kiskinova, "*Shedding light on electrodeposition dynamics tracked  *in situ* via soft X-ray coherent diffraction imaging*", *Nano Research*, Vol. **9**, Issue 7, (2016), pp. 2046-2056.
DOI: https://doi.org/10.1007/s12274-016-1095-9

[3] George Kourousias, Benedetto Bozzini, Michael W. M. Jones, Grant A. Van Riessen, Simone Dal Zilio, Fulvio Billé, Maya Kiskinova, and Alessandra Gianoncelli, "*Monitoring dynamic electrochemical processes with in-situ ptychography*", *Applied Nanoscience*, (2018), (*To appear*).
DOI: https://doi.org/10.1007/s13204-018-0703-2

## Developers
The software has been developed by the authors, members of the [Elettra Scientific Computing Team](https://www.elettra.trieste.it/it/lightsources/labs-and-services/scientific-computing/scientific-computing.html) and the [Image Processing Laboratory](https://www2.units.it/ipl/index.htm)  from the  [University of Trieste](https://www.units.it/).


## License

The project in under the GPL-v3 license.
