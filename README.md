# Quantum Machine Learning B-C-D in Qiskit
- **Author:** [Jacob Cybulski](https://jacobcybulski.com/) ([LinkedIn](https://www.linkedin.com/in/jacobcybulski/)), *Enquanted*
- **Associated with:** [QPoland](https://qworld.net/qpoland/)
- **Aims:** This is a workshop session introducing quantum machine learning for those already familiar with Quantum Computing algorithms and Qiskit.
- **Prerequisites:** This GitHub assumes good knowledge of quantum computing and machines learning, as well as previous experience with Python and Qiskit. 
- **Description:**
  This QML BCD lab explores the process of developing a simple quantum machine learning model in Qiskit.<br>
  The lab includes a practical session that covers the QML concepts, models, and techniques.<br>
  The initial lab tasks will be demonstrated by the presenter.<br>
  The following tasks are designed to be completed by the participants and discussed on Discord.
- **Creation Date:**<br>
  October 1, 2022: Initial development<br>
  September 20, 2025: Modified for audience of intermediate level of quantum computing proficiency.<br>
  September 27, 2025: Adopted for session "Building QML Models with Variational Quantum Algorithms", part of [QSilver-31 Workshop](https://qworld.net/qsilver31/), QBangladesh / QWorld.<br>
  October 10, 2025: Adopted for session "An introduction to Quantum Machine Learning in Qiskit", part of [Qiskit Fall Fest UAM 2025](https://research.ibm.com/events/qiskit-fall-fest-2025), [Adam Mickiewicz University](https://amu.edu.pl/en), Poznań, Poland.
- **How to cite this work:**<br>
  If you are a researcher and wanted to use these resources, please cite my work as follows.<br>

  _**The presentation (slides and video when available):**_<br>
  Jacob L. Cybulski, "Building QML Models with Variational Quantum Algorithms",<br>
  QSilver-31 Workshop by QBangladesh / QWorld, September 20-21 and 27-28, 2025,<br>
  [https://github.com/ironfrown/qml_bcd_lab](https://github.com/ironfrown/qml_bcd_lab).

  _**This GitHub:**_<br>
  Jacob L. Cybulski (ironfrown), "Quantum Machine Learning A-B-C in Qiskit",<br>
  GitHub, 2025,
  [https://github.com/ironfrown/qml_abc_lab/](https://github.com/ironfrown/qml_abc_lab/),
  Accessed Day-Month-Year.

## Folders
This site is structured as follows:
- _**dev:**_ developement versions of the three labs notebooks
- _**runs:**_ answers to all tasks set in the labs
- _**examples:**_ additional relevant QML examples
- _**utils:**_ utilities for charting, filing and segmenting data, a few models, etc.<br>
  In my demo, I will use mainly some clever chart plotting functions.
- _**slides:**_ presentation slides in PDF (as they become available)
- _**legacy:**_ previous versions of files (in case you really really wanted them)
  
## Important notebooks

You can play with these notebooks implementing QML time series analysis models in Qiskit, enjoy!<br>
Their simplicity is misleading as the tasks set by these labs will make you sweat.<br>
Note however that this GitHub is still in development and the notebooks may be added and updated at any time!

<table style="float: left;">
    <tr><th style="text-align: left;">Notebook</th>
        <th style="text-align: left;">Dataset</th>
        <th style="text-align: left;">Model</th>
        <th style="text-align: left;">Description</th>
    </tr>
    <tr><td style="vertical-align: top;"><strong><em>qml_bcd_01_mglass</em></strong></td>
        <td style="vertical-align: top;">mackie_glass</td>
        <td style="vertical-align: top;">qnn_tsa_model</td>
        <td style="vertical-align: top;">The exercise to create a Qiskit time series analysis model for Mackie-Glass data.</td>
    </tr>
</table><div style="clear: both;"></div>
            
## Requirements
This repository requires Qiskit v1.4.4 and Qiskit ML 0.8.3.<br>
at the development time Qiskit ML was incompatible with Qiskit v2.0+.

- Create a Qiskit conda environment and activate it<br>
    conda create -n qiskit-qml python=3.11<br>
    conda activate qiskit-qml
- install the Qiskit package, which includes ML:<br>
    pip install qiskit[visualization]==1.4.4<br>
    pip install qiskit-ibm-runtime<br>
    pip install qiskit-aer or pip install qiskit-aer-gpu<br>
    pip install qiskit-machine-learning ==0.8.3
- The following packages would have been installed as well:<br>
    numpy matplotlib pandas pillow rustworkx scipy scikit-learn seaborn
- Additional packages often used:<br>
    pip install jupyter jupyterlab <br>
    conda install freetype 
- Run jupyter or jupyter lab

All code was tested on Ubuntu 22.04-24.04.

## License
This project is licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html).<br>
The GPL v3 license requires attribution for modifications and derivatives,<br>
ensuring that users know which versions are changed and to protect the reputations of original authors.