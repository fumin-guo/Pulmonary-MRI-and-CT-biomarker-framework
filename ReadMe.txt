This package provides the main components of the pulmonary MRI and CT biomarker framework, including:

1) the software GUI (need Visual C++ Redistributable Packages for Visual Studio 2013 to run). 
   The source code and the .exe GUI-based 3D visualizer are not provided because of copyright issue. However, readers who are interested in the visualizer can contact 
	Dr. Aaron Fenster at: afenster@robarts.ca

2) three representative biomarker modules (the remainder will be added soon): 
	i) Whole lung and lobar noble gas MRI ventilation measurements
		Guo, Fumin, et al. "Globally optimal co-segmentation of three-dimensional pulmonary 1H and hyperpolarized 3He MRI with spatial consistence prior." Medical image analysis 23.1 (2015): 43-55.  
		Guo, Fumin, et al. "Thoracic CT-MRI coregistration for regional pulmonary structure-function measurements of obstructive lung disease." Medical physics 44.5 (2017): 1718-1733.
	
	ii) Multi-volume UTE MRI dynamic proton density maps
		Sheikh, Khadija, et al. "Ultrashort echo time MRI biomarkers of asthma." Journal of Magnetic Resonance Imaging 45.4 (2017): 1204-1215.
	
	iii) Inspiration and expiration CT (MRI) parametric response maps
		Capaldi, Dante PI, et al. "Pulmonary imaging biomarkers of gas trapping and emphysema in COPD: 3He MR imaging and CT parametric response maps." Radiology 279.2 (2016): 597-608

	iv) Fourier decomposition of free breathing 1H MRI ventilation measurements (not provided as this work is under review)


	v) 1H MRI specific ventilation measurements (not provided as this work is protected)
		Capaldi, Dante PI, et al. "Free-Breathing Pulmonary MR Imaging to Quantify Regional Ventilation." Radiology 287.2 (2018): 693-704.
	
3) test demos for each biomarker module and commonly used Matlab functionalities (these .m files needs to be added to Matlab path). 

Please refer to the "RunMe.txt" in each folder for specific components.
