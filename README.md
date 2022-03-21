# Quantification of corneal surgery antecedent
*Corentin Soubeiran, Maelle Vilbert, Anatole Chessel and Karsten Plamman*

## Abstract
We propose an algorithmic method generating morphological quantifiers of the corneal epithelium and in particular of Bowman's layer from corneal images acquired by optical coherence tomography (OCT). These quantifiers are calculated on OCT images of a cohort of 478 consenting patients at the Fifteen-Twenty Hospital in Paris. The cohort consists of 163 healthy patients and 315 with Fuchs' dystrophy. This cohort allowed the training of an artificial intelligence (Random Forest) for the classification of corneas into "healthy" and "pathological" classes resulting in a classification accuracy (or success rate) of 97%. The application of this artificial intelligence on images from patients having undergone Photo Keratectomy for Refractive Aiming (PKR) surgery shows that the use of the class probability provides a quantified indicator of the healing of the cornea in post-operative follow-up. The sensitivity of this probability was studied using repeatability data. Finally, the ability of artificial intelligence to detect subepithelial scars identified by clinicians as the origin of postoperative visual haze was demonstrated. 

## Introduction
The cornea fills two thirds of the optical power of the eye. Surgery such as Photo Keratectomy with Refractive Aim (PKR) allows the correction of visual acuity by correcting the shape of the stroma (under the corneal epithelium). In some patients, this surgery leads to the appearance of a sensation of "haze" due to sub-epithelial fibrosis, which gradually disappears with the healing process: this is called visual haze. To date, there are no diagnostic tools that provide information on the evolution of this scarring. Moreover, the detection of this haze is essentially based on feedback from the patient. There are thus few data where a haze is actually clinically validated. In addition, other pathologies such as Fusch's dystrophy can lead to the appearance of consequent subepithelial fibrosis, which are identified and monitored regularly in the clinic. Whether in postoperative follow-up of PRK surgery or in follow-up of clinical dystrophy, optical coherence tomography (OCT) is a routine examination.
The detection and identification of these fibroses is an essential element for the follow-up of the patient over time. A diagnostic tool providing digital information on these pathologies would allow a better interpretation of clinical examinations. On the other hand, this tool would also allow the detection of potential surgical histories on donor corneas in eye banks.  

### Corneal anatomy:
Histologically, the cornea consists of 3 layers that run parallel to each other. Each of these layers provides optical functions of refraction and transparency (ROTHSCHILD, n.d.)(Figure 1).
- Corneal epithelium: represents 10% of the total thickness of the cornea. It plays an essential optical role, but also a barrier function necessary for the protection of the eye. The stratified squamous epithelium is composed of three cellular layers. During refractive surgery with the PRK laser correction technique, the epithelium is removed from an area of about 8 to 9 mm in diameter. It takes 4 to 5 days to regrow.
- The Bowman layer: this acellular membrane separates the epithelium from the corneal stroma. Formed of collagen fibrils without orientation or periodicity, this layer has a thickness of 8 to 14 microns. During refractive surgery with the PRK laser, its circular peripheral section does not induce, a priori, any particular anomaly.
- The stroma: main layer of the cornea. It represents 90% of the cornea. The stroma is composed of collagen lamellae between which are corneal fibrocytes, also called keratocytes, and the fundamental substance.

### Optical Coherence Tomography:
Optical Coherence Tomography (OCT) is a modern ocular imaging procedure, which allows to obtain in a few seconds, and in a non-invasive way, images of the eye in section (Figure 1). It is a routine examination in patients for postoperative follow-up or for the follow-up of pathologies such as Fuchs' dystrophy. (Optical coherence tomography, 2019)

<figure>
<img src="Data\Figure_report\1_cornea.png">
<figcaption align = "center"><b>Figure 1: Anatomy of the cornea and corneal layers</b></figcaption>
</figure>

<figure>
<img src="Data\Figure_report\2_cornea_haze.png">
<figcaption align = "center"><b>Figure 2: OCT of a cornea after PKR surgery. A slight fibrosis of Bowman's layer is observed. Note that the central vertical line is an artifact of OCT acquisition. As well as the approximation of the Bowman's layer/fibrosis and the corneal surface in the centra</b></figcaption>
</figure>

### Photo Keratectomy for refractive purposes :
PRK is based on the delivery of a refractive correction to the surface of the corneal stroma, after removal of the superficial corneal epithelium. It does not require the cutting of a cap, unlike LASIK. It is performed under local anesthesia (drops). Both eyes are operated on the same day. There are two variants depending on how the epithelium is removed: in the "manual" technique, the epithelium is removed (peeled) with a sponge after application of a diluted solution in the "all-laser" or "transepithelial PRK" technique: the epithelium is photoablated by the excimer laser. (GATINEL, 2020)

### Visual haze or Haze :
During PKR, the epithelium is gently peeled after administration of anesthetic drops to numb the cornea: the superficial part of the stroma, called Bowman's layer, is then exposed to the excimer laser. The excimer laser beam, controlled by a computer coupled to the delivery system, is then projected onto the bare corneal surface to sculpt the superficial stromal corneal tissue. PRK induces a slightly longer healing phase than LASIK because of the time it takes for the epithelium to regrow on the surface of the reshaped cornea, which takes several days. Following this operation, a scar can be observed (Figure 2). Some patients report the appearance of a "visual haze" after surgery, also called haze. (GATINEL, 2020)

### Fuchs' dystrophy :
Fuchs' dystrophy is above all a pathology secondary to the aging of the corneal endothelium. It is an endothelial and epithelial corneal pathology. It is characterized by the histological elementary lesion and the formation of verrucosities of Descemet's membrane. An important element of the pathophysiology of the disease is the accelerated endothelial cell death by apoptosis. The advanced evolution of the dystrophy results in fibrosis, initially subepithelial. (BORDERIE Vincent, 2020)

### Database
- Healthy: 12 corneas of patients qualified as healthy by the ophthalmologist who follows them. 
- Healthy_2: 163 corneas of patients qualified as healthy by the ophthalmologist who follows them.
- Haze: 6 corneas of patients qualified as having "Haze" (scarring) by the ophthalmologist following them.
- Haze_2: 37 of 5 patients (10 eyes) before and after PRK surgery qualified as having a "Haze" (scarring veil) by the ophthalmologist who follows them after surgery.
- Fuchs: 315 corneas of patients qualified as having Fuchs' dystrophy by the ophthalmologist following them. 
- Repetability_healthy: 50 (10 patients, 1 eye/patient, 5 images/eye) corneas of patients qualified as healthy by the ophthalmologist following them. 
- Repetability_fuchs: 26 (3 patients, 2 eyes/patient, 5 to 6 images/eye) corneas of patients qualified as having Fuchs dystrophy by the ophthalmologist who follows them. 

## Characterization of the morphology of the subepithelial zone	
## Pre-processing of OCT images
The images acquired by OCT sometimes present artifacts due to the technology used:
- Central specular reflection: the curvature and reflexivity of the corneal surface generates, depending on the axis used by the manipulator during acquisition, the appearance of a vertical line of high intensity on the images (See Figure 2)
- An irregular exposure: the curvature of the cornea makes the intensity of the rays reflected on the surfaces uneven, in particular the peripheral zones present a reduced SNR. (See Figure 1, the central region presents a luminous halo and the peripheral regions are weakly exposed)

On the other hand, to carry out the study of Bowman's layer, it is necessary to flatten the cornea in order to ignore the curvature and thus compare the depths in relative terms (depth relative to the corneal surface). The processing chain is the one developed in (Vilbert, 2021). An illustration is given in Figure 3. (Bocheux R, 2019)

<figure>
<img src="Data\Figure_report\3_treatment_pipeline.png">
<figcaption align = "center"><b>Figure 3: OCT image processing to remove artefacts and correct exposure.</b></figcaption>
</figure>

<figure>
<img src="Data\Figure_report\4_extraction_bell_profile.png">
<figcaption align = "center"><b>Figure 4: Extraction of Bowman's layer profile (bell profile).</b></figcaption>
</figure>

<figure>
<img src="Data\Figure_report\5_quantification.png">
<figcaption align = "center"><b>Figure 5: Some quantifiers illustrated. Area ratio correspond to the ratio of the orange area (excluding peak) and blue area including the peak.</b></figcaption>
</figure>

## Characterization on average profile

Via the images of flattened corneas, we extract the average axial profile of the cornea. This profile shows a first peak corresponding to the surface of the cornea, a second peak corresponding to Bowman's layer (for healthy corneas). After this second peak, the stroma appears with a decreasing exponential profile (studied for the development of transparency quantifiers in (Bocheux R, 2019) (Vilbert, 2021)) (Figure 4 A,B,C). 
From the mean axial profile, the peak related to Bowman's layer (in the case of healthy corneas) or subepithelial fibrosis is identified by considering the local minima around the peak, and bounded by the minimum distance between the position of the peak and the minima on both sides (Figure 4 D). On this profile we perform a Gaussian regression of the profile (after an interpolation phase to increase the data and ensure better regression results). The parameters of the Gaussian regression provide 4 quantifiers: 
- Position of the mean: " mean " , then translating into depth relative to the surface in μm
- Standard deviation: " sigma ", qualifying the spread of the peak or thickness of the layer in μm
- Distance between the two local minima: " peak width ", also qualifying the spread of the peak or thickness of the detected layer but before the regression (in μm).
- The intensity of the peak: "intensity peak", corresponding to the intensity normalized to the rest of the peak profile.

From the average profile, we also perform an exponential regression between the first peak and the end of the stroma to characterize the intensity decay of the profile. This regression provides two quantifiers corresponding to the fitting variables of the regression function:
$$f(x)=α.e^{-βx}+C $$
We thus isolate:
- The amplitude: "β"
- The decay rate: "α"

We also create a quantizer of the optical contrast. For this we calculate the area under the curve of the signal with and without the peak considered. The quotient of these two quantities then provides an indication of the optical contrast contained in the peak. Hence the quantifier : 
- Area ratio: "area ratio" illustrated in Figure 5.
To this we add two parameters qualifying the quality of the regression:
- The mean square error: "MSE", characterizing the error committed by the Gaussian regression with respect to the real profile.
- Covariance between the profile and the regression: "DataCov", characterizing the joint deviation from the respective expectations of the profile and the Gaussian.
These last two parameters were introduced following the observation of the images. While on healthy corneas the Gaussian regression is often well resolved, it is not always so on pathological corneas: a bad squared error or covariance can be an indicator of pathology.

### Calculation and dispersion of parameters	
We have defined 9 quantifiers (Figure 6) of the cornea specialized in the evaluation of Bowman's layer or subepithelial fibrosis in pathological corneas. The calculation of these parameters is performed by a Python algorithm. We have built a class " image_Quantification.py " which performs all the processing on a pre-processed image. The initial processing of the images is also done by a specific class "image_OCT.py". This class architecture allows us to quickly process a large number of images in the form of objects and their attributes. Moreover, we can access all processing results on different objects simultaneously. 

By using this algorithm on all the images of a database we can build a python object "dataframe" integrating the values of each quantifier for each image.

We apply this methodology to the 7 databases presented previously and obtain the dataframes:
- data_healthy: 12 corneas of patients qualified as healthy by the ophthalmologist who follows them. 
- data_healthy_2 : 163 corneas of patients qualified as healthy by the ophthalmologist who follows them.
- data_haze : 6 corneas of patients qualified as having a "Haze" (scarring veil) by the ophthalmologist who follows them.
- data_haze_2: 37 of 5 patients (10 eyes) before and after PRK surgery qualified as having Haze by the ophthalmologist following them after surgery.
- data_fuchs: 315 corneas of patients qualified as having Fuchs' dystrophy by the ophthalmologist following them. 
- data_rep_healthy: 50 (10 patients, 1 eye/patient, 5 images/eye) corneas of patients qualified as healthy by the ophthalmologist following them. 
- data_rep_fuchs: 26 (3 patients, 2 eyes/patients, 5 to 6 images/eye) corneas of patients qualified as having Fuchs' dystrophy by the ophthalmologist following them. 
These dataframes are saved in pkl format in the dataset folder.
 We present in Figure 7 the parameter distributions for data_healthy_2 and data_fuschs. For a large part of the quantifiers, the distributions are statistically significantly different (very low p values). For alpha, beta, intensity_peak and dataCov, the difference between the distributions is less obvious.

<figure>
<img src="Data\Figure_report\6_processing_flow.png">
<figcaption align = "center"><b>Figure 6: Data processing flow chart for the creation of the quantifiers. (U=normalization between 0 and 1).</b></figcaption>
</figure>

<figure>
<img src="Data\Figure_report\quantifiers_distrib.png">
<figcaption align = "center"><b>Figure 7: Distribution of the 9 parameters for data_healthy_2 and data_fuschs datasets, p values corresponding to t.test with welch..</b></figcaption>
</figure>

## Classification model
Based on the quantified data described above, we seek a method for classification of healthy corneas and patients with Fuchs' dystrophy. 
### Available data
The databases of post-PKR surgery at our disposal presenting visual haze/haze is very limited. An initial "one class SVM" approach (Xiyan He, 2011) proved unsatisfactory in correctly characterizing healthy corneas with sufficient accuracy (pathologic corneas were classified as healthy with more than a 99% chance). 

Since we have a large number of images with Fuchs' dystrophy, since in this pathology subepithelial fibrosis can occur, and since scarring/haze is characterized as a case of epithelial fibrosis, we propose to use corneas with Fuchs' dystrophy as a way to distinguish pathological corneas from healthy corneas in certain measures. 

Classical classifiers usually provide a classification probability. This probability generally corresponds to a degree of confidence that the classifier evaluates for each class when processing data.

### Learning
We created a dataframe extracted from data_healthy_2 and data_fuchs combining the 9 quantifiers as a feature (input), the value to be predicted (output) being the class (1-positive: for corneas with Fuchs dystrophy, and 0-negative for healthy corneas).  This dataframe is separated into 2 sets, a training set (80%) and a test set (20%) separated randomly. On the training set we perform a cross-validation technique. 

Cross-validation is a statistical method that allows to evaluate the performance of machine learning models. In our case, it consists in separating the training set into 10 pieces, iteratively training the model on 9 pieces, and evaluating its performance on the 10th piece. We average the different success rates (accuracy). This approach gives an idea of the accuracy (or success rate) of the classification in the general case and does not favor an over-trained model.

<figure>
<img src="Data\Figure_report\8_confusion_matrix_test.png">
<figcaption align = "center"><b>Figure 8: Confusion matrix on test data with random forest model.</b></figcaption>
</figure>

<figure>
<img src="Data\Figure_report\9_missclass1.png">
<img src="Data\Figure_report\9_missclass2.png">
<figcaption align = "center"><b>Figure 9: Missclassified images by the random forest quantifier..</b></figcaption>
</figure>

## Results
The results of the different models are given in Table 1. The best performing method seems to be the random forest. This classification method, also called decision tree forests, is based on decision tree learning, with training on multiple decision trees trained on slightly different subsets of the data. The results shown in Table 1 correspond to a fitting of the model hyperparameters (max sample, max features, number of trees, tree depth) the model is saved as a pkl file "RF_model.pkl".
This model is then used to process the test set (for the first and only time to ensure a successful generalization rate). We then obtain a rate of 97.0%.  The confusion matrix is presented in Figure 8. The model therefore makes two errors. The images on which these errors are made are shown in Figure 9. On these images, we give the probability displayed by the model for the class considered. We notice that in both cases the classification is not "clear cut", with probabilities around 50%.

To evaluate these two images, we asked 5 ophthalmologists to rate them on a scale of 0 (pathological cornea) to 10 (healthy cornea) without context and without consultation. For the first image (Figure 9 top) the doctors returned a mean score of 0.6/10 (standard deviation 0.8) unambiguously judging the cornea as pathological (like its label) contrary to the model. For the second image (Figure 9 bottom), with a mean score of 2.6/10 (standard deviation 2.42), the cornea is also judged rather pathological contrary to its label. There is thus an ambiguity that could justify the error of the model.

Using the two databases used for training data_healthy_2 and data_fuchs, we generate Figure 11 representing the model probability distributions for the two classes. We notice that the probability averages of these two classes are significantly different, justifying the rarity of ambiguities and the relevance of focusing on the few contentious misclassified images that remain isolated cases.

<figure>
<img src="Data\Figure_report\tableau1.png">
<figcaption align = "center"><b>Table 1: Classification accuracy for different models.</b></figcaption>
</figure>

<figure>
<img src="Data\Figure_report\10_repetability_proba.png">
<figcaption align = "center"><b>Figure 10: Results of repeatability on corneas qualified with Fuchs dystrophy (A) on healthy corneas (B). For A and B dots size correspond to SNR (smaller the dot is better the SNR). (C): statistical repartition of the probability.</b></figcaption>
</figure>

<figure>
<img src="Data\Figure_report\11_probability_distribution_alldataset.png">
<figcaption align = "center"><b>Figure 11: dispersion of class probability computed by the model on the test and train data. (left) histogram. ((Right) statistical dispersion (boxplot).</b></figcaption>
</figure>

## Repeatability

In order to evaluate the robustness of the classification we perform a repeatability study. In this study 13 patients are selected by an ophthalmologist as having healthy corneas (10 patients) or with Fuchs dystrophy (3 patients). On these patients we perform 5 OCT acquisitions by 5 different manipulators. These images are then presented to the classification model. This results in an accuracy of 100% on the classification (Figure 10: A and B). The statistical distribution is given in Figure 10 C, we notice that the standard deviation is worse (0.09) for healthy corneas than for pathological corneas (0.05) in this study. We did not identify any particular correlation between SNR quality and the dispersion of images from the same patient.

## Confrontation with post-PKR corneas
As the origin of this classification remains the search for an identification of post-surgical history, we now move on to databases of corneas that have or may have a scar/haze.
### Identification of scar/haze veils
We submit images from two databases initially provided by ophthalmologists, consisting of 6 corneas with post-PKR (refractive photokeratectomy) scar/haze and 12 healthy corneas (haze and healthy databases). The results are shown in Figure 12. All pathological corneas are classified as such, one healthy cornea among the 12 is misclassified (as pathological) with a probability of 51%. Again, the probability is very close to 50% and therefore ambiguous. 

We asked the cohort of 5 ophthalmologists to judge this image. They gave it a score of 9/10 (standard deviation of 1.09), thus classifying it as healthy as its label. The ambiguity of the model thus seems unjustified from a clinical point of view.

<figure>
<img src="Data\Figure_report\12_haze_vs_healthy.png">
<figcaption align = "center"><b>Figure 12: Model probability dispersion for haze and healthy dataset .</b></figcaption>
</figure>
### Post-surgical follow-up
We have at our disposal a small set of images composed of 5 patients who underwent PKR surgery (haze_2). This is an operation aimed at correcting visual acuity. In our study, these treated corneas can be pathological or healthy. From a clinical point of view, some corneas operated by PRK develop an inflammatory scarring reaction, which manifests itself by a haze disturbing the vision of the patients. The healing process can last from 3 to 7 days, depending on the patient. For some patients, recovery is longer, of the order of several weeks. After this period, the cornea should gradually return to normal parameters. 

These images are presented to the model and we study the evolution of the classification probability before and after surgery. We notice two main regimes: a first regime of corneas classified as healthy, which after surgery are classified as pathological and then healthy again. A second regime of corneas classified as pathological before and just after surgery and then classified again as healthy. (Figure 13, Figure 14)
The analysis of the importance of the parameters in the classification (Figure 15) shows that "sigma" is the most important quantifier. This parameter corresponds to an estimator of the thickness of Bowman's layer or subepithelial fibrosis in pathological cases. In terms of distribution (Figure 7), it was already a parameter leading to a statistically significant difference in dispersion. The specific study of this parameter for repeatability shows less marked results than those obtained on the classification probability (in Figure 17). Applied on the small dataset "visual haze" vs. "healthy" (healthy and haze), the values of the two datasets remain separable. 

Finally the study of the postoperative evolution of this parameter seems to indicate a convergence over time towards a value of 10 μm (Figure 18). Using a scale relative to the initial value, we again notice the appearance of two regimes: one for initially high sigma values that tend to decrease and one for initially low values that tend to increase. These results are very similar to the results of probability analyses (Figure 16). We conclude then, as expected when analyzing the main parameters of the random forest, that the classifier gives a result very related to this parameter, thus providing interpretability to its predictions. Note that training on only the sigma parameter leads to a validation accuracy of only 92.7%, so the additional parameters added seem to increase the accuracy to 97%. 

<figure>
<img src="Data\Figure_report\13_post_op_pb.png">
<figcaption align = "center"><b>Figure 13: (Left) evolution of the model probability for cornea of patients before (-1) just after (0) and after PKR surgery. (Right) same data adjusted to make the stating probability equal to 0.</b></figcaption>
</figure>
<figure>
<img src="Data\Figure_report\14_post_op_pb_fromhealthy.png">
<img src="Data\Figure_report\14_post_op_pb_fromPatho.png">
<figcaption align = "center"><b>Figure 14: Example of classification probability for two patients. After surgery (Month>0) the cornea is classified as pathological. With time the model classifies the corneas as healthy with more and more confidence. .</b></figcaption>
</figure>
<figure>
<img src="Data\Figure_report\15_Figure_importance.png">
<figcaption align = "center"><b>Figure 15: Feature importance for our Random forest model.</b></figcaption>
</figure>
<figure>
<img src="Data\Figure_report\16_haze_vs_healthy_sigma.png">
<figcaption align = "center"><b>Figure 16: Sigma values dispersion for haze and healthy dataset.</b></figcaption>
</figure>
<figure>
<img src="Data\Figure_report\17_repetability_sigma.png">
<figcaption align = "center"><b>Figure 17: Repeatability of the parameter sigma. (A): On healthy control patients. (B): On patients with Fuchs dystrophy. (C) Statistical dispersion..</b></figcaption>
</figure>
<figure>
<img src="Data\Figure_report\18_post_op_sigma.png">
<figcaption align = "center"><b>Figure 18: Post-op evolution of parameter sigma through time. (Left) value of the parameter in μm. (Right) Relative value compared to initial value.</b></figcaption>
</figure>

## Conclusion
We have established an algorithm for the classification of healthy and pathological corneas from the perspective of subepithelial fibrosis. This algorithm specifically targets morphological features of Bowman's layer or subepithelial fibrosis. We have shown that applying a classification model to these morphological quantifiers leads to valid predictions in 97% of cases, while relying mainly on a thickness parameter providing an interpretable and measurable criterion to the clinician. This set of algorithms thus shows promising results in the study of the evolution of certain corneal injuries in postoperative follow-up.

## Discussion
It is unfortunate that this study involved only a limited number of patients for temporal follow-up (5 patients, 10 eyes). Increasing the number of patients should allow the evaluation to be validated or invalidated. Moreover, the reduced frequency of acquisition does not allow to appreciate the transitions between a healthy and pathological state (and vice versa). Some data with a weekly or even daily acquisition would allow to evaluate a correlation between class probability and time. 
