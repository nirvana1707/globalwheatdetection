# Global Wheat Detection
This project was created as part of a Kaggle research competition with this title. 
https://www.kaggle.com/c/global-wheat-detection
The objective of the the competition was to detect "wheat heads" on plan containing grain.
The competition required participants to Computer Vision based pipeline to detect wheat heads from out door images of wheat plants.

# Background
Images are wheat fields are used to estimate the density and size of spikes which are used by farmers to assess health and maturity when making decisions on field. 

# Challenges
Detection of wheat heads often encounter a variety of challenges raning from climatic, geographic and phenotypic conditions. Some of which are :
a.) overlap of dense wheat plants
b.) climatic conditions such as wind, rain blurring photograps
c.) unwanted objects (such as insects) sitting atop the wheat heads
d.) variation in appearance on account of maturity, color, genotype, and orientation
e.) geographical variations in varieties, densities, pattens and overall field/environment conditions.

Current detection involve one and two stage detectors (Yolo-V3 and Faster RCNN), but when trained with a large dataset, a bias to the training region remains.

# Dataset
Dataset for the challenge has been provided by Kaggle. However, the data collation was led by nine research institutes from seven countries: the University of Tokyo, Institut national de recherche pour l’agriculture, l’alimentation et l’environnement, Arvalis, ETHZ, University of Saskatchewan, University of Queensland, Nanjing Agricultural University, and Rothamsted Research. These institutions are joined by many in their pursuit of accurate wheat head detection, including the Global Institute for Food Security, DigitAg, Kubota, and Hiphen.


