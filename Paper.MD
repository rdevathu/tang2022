## Circulation: Arrhythmia and Electrophysiology

```
Circulation: Arrhythmia and Electrophysiology is available at http://www.ahajournals.org/journal/circep
```
```
Correspondence to: Tina Baykaner, MD, MPH, Stanford University, 453 Quarry Rd, 334C, Palo Alto, CA 94304. Email tina4@stanford.edu
The abstract was presented at Heart Rhythm Society (HRS) Scientific Sessions. The abstract embargo for this meeting will be lifted on April 28th, 2022.
This article was sent to N.A. Mark Estes III, MD, Guest Editor, for review by expert referees, editorial decision, and final disposition.
Supplemental Material is available at https://www.ahajournals.org/doi/suppl/10.1161/CIRCEP.122.010850.
For Sources of Funding and Disclosures, see page 508.
© 2022 American Heart Association, Inc.
```
## ORIGINAL ARTICLE

## Machine Learning–Enabled Multimodal Fusion

## of Intra-Atrial and Body Surface Signals in

## Prediction of Atrial Fibrillation Ablation Outcomes

#### Siyi Tang , MS; Orod Razeghi , PhD; Ridhima Kapoor , MD; Mahmood I. Alhusseini, MS; Muhammad Fazal, MD;

#### Albert J. Rogers , MD, MBA; Miguel Rodrigo Bort, PhD; Paul Clopton , MS; Paul J. Wang , MD; Daniel L. Rubin , MD;

#### Sanjiv M. Narayan , MD, PhD; Tina Baykaner , MD, MPH

#### BACKGROUND: Machine learning is a promising approach to personalize atrial fibrillation management strategies for

#### patients after catheter ablation. Prior atrial fibrillation ablation outcome prediction studies applied classical machine

#### learning methods to hand-crafted clinical scores, and none have leveraged intracardiac electrograms or 12-lead surface

#### electrocardiograms for outcome prediction. We hypothesized that (1) machine learning models trained on electrograms

#### or electrocardiogram (ECG) signals can perform better at predicting patient outcomes after atrial fibrillation ablation

#### than existing clinical scores and (2) multimodal fusion of electrogram, ECG, and clinical features can further improve the

#### prediction of patient outcomes.

#### METHODS: Consecutive patients who underwent catheter ablation between 2015 and 2017 with panoramic left atrial

#### electrogram before ablation and clinical follow-up for at least 1 year following ablation were included. Convolutional neural

#### network and a novel multimodal fusion framework were developed for predicting 1-year atrial fibrillation recurrence after

#### catheter ablation from electrogram, ECG signals, and clinical features. The models were trained and validated using 10-fold

#### cross-validation on patient-level splits.

#### RESULTS: One hundred fifty-six patients (64.5±10.5 years, 74% male, 42% paroxysmal) were analyzed. Using electrogram

#### signals alone, the convolutional neural network achieved an area under the receiver operating characteristics curve (AUROC)

#### of 0.731, outperforming the existing APPLE scores (AUROC=0.644) and CHA2DS2-VASc scores (AUROC=0.650).

#### Similarly using 12-lead ECG alone, the convolutional neural network achieved an AUROC of 0.767. Combining electrogram,

#### ECG, and clinical features, the fusion model achieved an AUROC of 0.859, outperforming single and dual modality models.

#### CONCLUSIONS: Deep neural networks trained on electrogram or ECG signals improved the prediction of catheter ablation outcome

#### compared with existing clinical scores, and fusion of electrogram, ECG, and clinical features further improved the prediction. This

#### suggests the promise of using machine learning to help treatment planning for patients after catheter ablation.

#### GRAPHIC ABSTRACT: A graphic abstract is available for this article.

##### Key Words: Machine learning ◼ atrial fibrillation ◼ cardiac electrophysiology ◼ catheter ablation

# A

#### trial fibrillation (AF) ablation is the cornerstone of

#### therapy for symptomatic AF, and it helps improve

#### quality of life and prolongs survival in several pop-

#### ulations.1,2 Improved tools for predicting the success

#### of AF catheter ablation are needed to guide clinicians

#### in better patient selection for this procedure, as well

#### as setting realistic patient expectations following the

#### procedure.

###### Downloaded from http://ahajournals.org by on September 18, 2025


#### Clinical scores have been developed to predict suc-

#### cess after catheter ablation of AF with area under

#### the receiver operating characteristics curve (AUROC)

#### of 0.55 to 0.65 for majority of the models, with rare

#### models reaching an AUROC of 0.75.3–5 However, none

#### of these previous predictive scores have incorporated

#### electrophysiological data, which may place specific AF

#### mechanisms within the clinical context to improve pre-

#### dictive accuracy.

#### We hypothesized that (1) machine learning (ML)

#### models trained on intracardiac electrograms or surface

#### electrocardiograms (ECG) signals can perform better

#### at predicting patient outcomes after AF ablation (ie,

#### 1-year AF recurrence) compared with existing clinical

#### scores and (2) multimodal fusion of electrogram, ECG,

#### and clinical features can further improve the prediction

#### of patient outcomes.

#### Although there are no prior ML-based studies that

#### directly take signals as inputs to predict AF ablation out-

#### comes, recent advances in the use of ML in signal analysis

#### of human rhythm disorders have led to promising prelimi-

#### nary results. For example, ML models were able to pre-

#### dict future ventricular arrhythmia from ventricular signals.^6

#### Prior works using ML to predict success of AF ablation

#### includes estimation of recurrence by predicting shape

#### descriptors directly from magnetic resonance imaging^7

#### and combining imaging and clinical biomarkers to predict

#### cryoballoon pulmonary vein isolation (PVI) outcomes.^8 ML

#### methods and personalized computational modeling have

#### also been used together to predict recurrence following

#### PVI.^9 In addition, handcrafted features derived from com-

#### puterized tomography (CT) scans have been shown to be

#### associated with likelihood of postablation AF recurrence.^10

#### Deep neural networks are the state-of-the-art ML

#### models that are able to learn complex features directly

#### from large amounts of data without the need of feature

#### engineering.^11 Deep neural networks have shown promis-

#### ing empirical successes across a wide variety of medi-

#### cal domains.^12 Unlike previous works using classical ML

#### models,8–10 we aim to develop and validate (1) a deep

#### neural network for post-ablation AF recurrence prediction

#### from signals (electrogram and ECG) and (2) a multimodal

#### fusion framework that leverages the three modalities––

#### electrogram, ECG, and patients’ clinical features––to fur-

#### ther improve the model performance (Figure 1A).

## METHODS

##### The data that support the findings of this study are available

##### from the corresponding author upon reasonable request.

### Subject Recruitment

##### This is a retrospective analysis of consecutive adult patients

##### with paroxysmal or persistent AF who underwent catheter

##### ablation between 2015 and 2017 at a tertiary referral center

##### by 5 providers. To be included, patients were required to have

##### panoramic left atrial electrograms recorded before ablation

##### and clinical follow-up for at least 12 months following abla-

##### tion for accurate assessment of their AF ablation procedure

##### outcomes. All patients had pulmonary vein isolation as a part

##### of the AF ablation procedure; additional ablation lesions per

##### the operating physicians’ discretion were allowed. This com-

##### prised of ablation of localized AF sources via focal impulse and

##### rotor mapping (FIRM, in 100% of patients), ablation of left atrial

##### linear lesions (in 24% of patients) and cavotricuspid isthmus

##### ablation (in 27% of patients).

## Nonstandard Abbreviations and Acronyms

#### AAD antiarrhythmic drug

#### AF atrial fibrillation

#### AU ROC area under the receiver operating char-

#### acteristics curve

#### CAD coronary artery disease

#### CatBoost categorical boosting classifier

#### CKD chronic kidney disease

#### CNN convolutional neural network

#### CT computerized tomography

#### LA left atrium

#### Lad left atrial diameter

#### ML machine learning

#### PVI pulmonary vein isolation

### WHAT IS KNOWN?

- Atrial fibrillation ablation is the cornerstone of ther-

#### apy for symptomatic atrial fibrillation, with increas-

#### ing evidence on its safety and efficacy.

- Clinical scores have been developed to predict

#### success of catheter ablation, to guide better patient

#### selection, with most clinical scores reaching an area

#### under the receiver operating characteristics curve

#### (AUROC) of 0.55 to 0.65 in accurately predicting

#### atrial fibrillation ablation success.

### WHAT THE STUDY ADDS

- Deep neural networks trained on intracardiac sig-

#### nals and 12-lead electrocardiogram signals, in addi-

#### tion to clinical features, can improve the prediction

#### accuracy of catheter ablation outcomes compared

#### with existing clinical scores.

- A convolutional neural network using intracardiac

#### signals in atrial fibrillation achieves an AUROC

#### of 0.731, similarly a convolutional neural network

#### using 12-lead electrocardiogram alone achieves an

#### AUROC of 0.767. Fusion of electrogram, electrocar-

#### diogram, and clinical features further improves the

#### prediction (AUROC=0.859) compared with models

#### with a single modality.

- Machine learning models can help treatment plan-

#### ning for patients after catheter ablation of atrial

#### fibrillation through more accurate prediction of

#### Downloaded from http://ahajournals.org by on September 18, 2025treatment outcomes.


##### Clinical and demographic data were obtained from elec-

##### tronic medical records. Twelve-lead ECGs in sinus rhythm

##### obtained within 1 year of the ablation procedure were included.

##### Patients with no 12-lead ECG available (n=3) were excluded

##### from the ECG-only model and were imputed with the means

##### of the other patients’ ECG features in the fusion models. This

##### study protocol was approved by the Institutional Review Board

##### of Stanford University. Due to the retrospective nature of the

##### study, no informed consent was required. The corresponding

##### author had full access to all the data in the study and took

##### responsibility for its integrity and the data analysis.

### Ablation Procedure and Clinical Follow-Up

##### All procedures were performed under general anesthesia. Various

##### ablation catheters were used to achieve PVI, which included point-

##### by-point radiofrequency ablation with a contact force sensing 3.

##### mm tip irrigated catheter (Biosense Webster; Abbott) or cryobal-

##### loon (Arctic Front, Medtronic). Unipolar panoramic intracardiac

##### signals used for ML analysis were obtained before any ablation

##### with a 64-pole basket catheter (FIRMap catheter, Abbott) during

##### AF. If patients presented to the electrophysiology laboratory in

##### normal sinus rhythm, AF was induced with burst pacing.

###### Figure 1. Overview of our methods and multimodal fusion framework.

###### A , Overview of our methods. The inputs come from 3 modalities: patient electrogram (EGM) signals, electrocardiogram (ECG) signals, and

###### clinical features. A multimodal machine learning model fuses the inputs from the 3 modalities and outputs prediction of atrial fibrillation (AF)

###### recurrence. B , Details of our multimodal fusion framework. We first trained a model on EGM signals only for AF recurrence prediction, and

###### a separate model on ECG signals only for AF recurrence prediction. We then extracted EGM and ECG features from the respective trained

###### models. Finally, the EGM and ECG features were concatenated with the clinical features and were subsequently passed to a multimodal fusion

###### model to predict AF recurrence.

###### Downloaded from http://ahajournals.org by on September 18, 2025


##### Patients were followed up routinely in the outpatient set-

##### ting, and all had 3-month evaluations for at least 1 year, which

##### included rhythm assessment with 12-lead ECGs at 3 and 6

##### months and a 14-day event monitor at 1 year. AF recur-

##### rence was defined as >30 second duration episodes on ECG

##### monitoring, or >1% AF burden on device interrogation for the

##### patients with implantable monitors. In this study, we focus on

##### the outcome of whether a patient has recurrent AF within 1

##### year after catheter ablation.

### Demographic and Clinical Features

##### The demographic variables extracted from electronic health

##### records included patients’ age at the time of ablation, sex, height,

##### weight, body mass index, race, and ethnicity. Clinical comorbidi-

##### ties such as presence of hypertension, hyperlipidemia, transient

##### ischemic attack, stroke (CVA), coronary artery disease, diabe-

##### tes (DM), chronic kidney disease, congestive heart failure, and

##### obstructive sleep apnea were collected. Arrhythmia character-

##### istics such as type of AF (paroxysmal, persistent or long stand-

##### ing persistent), and history of prior AF ablation were recorded.

##### Structural features extracted from imaging studies included

##### left ventricular ejection fraction and left atrial diameter from

##### transthoracic echocardiograms; and left atrial volume, surface

##### area and sphericity index from CT scans that were routinely

##### obtained within 1 year before AF ablation. These variables were

##### selected based on the literature on known factors which could

##### impact AF ablation outcomes.3–5,8,13–16 A complete list of clini-

##### cal features and number of missing values is shown in Table

##### S1. Missing values were imputed with the most frequent value

##### of the feature. Table S2 shows model performance with differ-

##### ent missing value imputation techniques, and Table S3 shows

##### model performance in patients without missing values.

### Modeling Clinical Features for AF Recurrence

### Prediction

##### As a baseline method, we built a classifier for predicting 1-year

##### AF recurrence from demographic and clinical features. For each

##### patient, a multi-dimensional feature vector was constructed

##### from the clinical and demographic features, where continuous

##### variables were normalized to have zero mean and unit variance

##### and categorical variables were one-hot encoded. We used the

##### categorical boosting (CatBoost) classifier,^17 a state-of-the-art,

##### gradient boosted decision tree-based ML algorithm, for AF

##### recurrence prediction. Briefly, CatBoost sequentially builds

##### many weak learners (ie, decision trees) and creates a strong

##### predictive model by greedy search and ensembling. We chose

##### CatBoost because it has been shown to outperform other gra-

##### dient boosted decision tree-based algorithms and naturally

##### handles both continuous and categorical variables.^17

### Preprocessing of Electrogram and ECG Signals

##### In each patient, unipolar left atrial intracardiac electrograms

##### were recorded during AF. Unipolar signals were recorded from

##### a 64-pole basket catheter positioned in the mid left atrium (LA)

##### before any ablation were exported. Preprocessing of electro-

##### gram signals included QRS subtraction and resampling to 200

##### Hz. See Supplemental Methods for details.

##### Preprocessing of ECG signals included a bandpass fil-

##### tering of 0.05 to 100 Hz and resampling to 200 Hz. Eight

##### independent ECG channels were used (channels I, II, and

##### V1-6) as any linear dependency can be naturally learned by

##### deep neural networks (ie, channel III can be derived vectorially

##### from channels I and II).

##### Each electrogram and ECG signal was augmented by divid-

##### ing into 5-sec windows with a 4-sec overlap between consecu-

##### tive windows, resulting in a 1000×64 matrix for each input

##### electrogram data point and a 1000×8 matrix for each input

##### ECG data point.

### Modeling Electrogram and ECG Signals for AF

### Recurrence Prediction

##### We developed a convolutional neural network (CNN) for pre-

##### dicting 1-year AF recurrence from electrogram or ECG signals.

##### Similar to Attia et al,^14 our CNN consisted of several lay-

##### ers of bottleneck blocks with 1-dimensional (1D) convolutions

##### operating on the time dimension, followed by a 1D convolu-

##### tional layer operating on the channel dimension. Intuitively,

##### the time-dimension convolutional layers capture the temporal

##### dependency in the signal by extracting features from signals

##### within one channel, whereas the final channel-dimension con-

##### volutional layer aggregates the features across channels to

##### obtain a spatial representation of the signal. Details of the CNN

##### can be found in Supplemental Methods and Figure S1.

### Fusion Model for AF Recurrence Prediction

##### Finally, we developed a multimodal fusion framework that lever-

##### ages more than one modality to improve the prediction of AF

##### recurrence (Figure 1B).

##### First, electrogram features were extracted from the CNN

##### that was trained on electrogram signals only. All the features

##### from the electrogram signals from the same patient were aver-

##### aged to obtain a single electrogram feature representation for

##### each patient. ECG features for each patient were extracted

##### in a similar way. Next, for each patient, the feature vectors of

##### the fused modalities (ie, electrogram features, ECG features,

##### and clinical features) were concatenated to form a multimodal

##### feature vector. Last, a classifier was trained on the patients’

##### multimodal feature vectors for predicting 1-year AF recurrence.

##### As a fair comparison to clinical feature-based models, we also

##### applied the CatBoost^17 classifier in the fusion framework.

##### As ablation experiments, we also validated fusion of 2

##### modalities (ie, electrogram and clinical features, ECG and clini-

##### cal features, or electrogram and ECG features) and compared

##### the results to fusion of 3 modalities (electrogram, ECG, and

##### clinical features).

### Model Training and Validation

##### Stratified 10-fold cross-validation (patient-wise split) was used

##### to train and validate each of the models described above.

##### Specifically, all patients were randomly divided into 10 groups (ie,

##### folds) with the same proportion of AF recurrence in each fold (ie,

##### stratified 10-fold). At the i-th cross-validation step, the i-th fold

##### was used to test the model and the remaining 9 folds were used

##### to train the model. This above process was repeated 10 times,

##### such that each patient only appeared in one of the test folds.

##### To mitigate overfitting, data augmentation was applied dur-

##### ing training. We designed 5 data augmentation methods using

##### electrophysiology domain knowledge: (1) randomly shift (forward

###### Downloaded from http://ahajournals.org by on September 18, 2025


##### or backward in time) each 5-sec window by up to 2.5-sec, (2)

##### randomly scale the raw signal by a factor within range 0.5 to 2,

##### (3) randomly shift the DC value within range −10 to 10 micro-

##### volts, (4) randomly masking with zeros for up to 25% of the 5-sec

##### window, (5) randomly add gaussian noise with zero mean and a

##### SD<0.2. Importantly, these data augmentations did not result in

##### invalid signals but naturally increased the variability of the training

##### data, which could mitigate overfitting of deep neural networks.

##### Training for the CNNs on electrogram and ECG signals

##### was accomplished using the Adam optimizer^18 in PyTorch on

##### a single NVIDIA P100 GPU. For CNNs, we followed the same

##### model architecture configuration as that in Attia et al^14 (except

##### for reducing the number of bottleneck blocks from 9 to 6 in

##### ECG-based CNN) and did not tune the model hyperparameters.

##### Training for CatBoost was done using the CatBoost Python

##### package,^17 and CatBoost hyperparameters were tuned using

##### grid search (see Supplemental Methods for details). All mod-

##### els were trained to optimize AUROC. We assessed the model’s

##### ability to predict 1-year AF recurrence using AUROC, sensi-

##### tivity, specificity, accuracy, and F1-scores. To derive sensitivity,

##### specificity, accuracy, and F1-scores, a probability threshold was

##### selected based on the highest F1-score on the 10 fold test sets.

### Statistical Analysis

##### For population characteristics, continuous data are reported as

##### mean±SD, unless otherwise stated, and are tested for normal-

##### ity using the Shapiro-Wilk test ( P >0.05). Independent samples

##### t test and Mann-Whitney U test were run to determine if there

##### were differences in mean values between cohorts for analy-

##### sis of continuous data. Categorical variables were compared

##### using the Pearson χ^2 test or Fisher exact test where expected

##### frequencies were <5. For model evaluation, we report the

##### mean and SD of AUROC, sensitivity, specificity, accuracy, and

##### F1-scores of the 10-fold test results. In addition, we measure

##### the calibration of the models using Brier score^19 and expected

##### calibration error (ECE).^20 Briefly, the Brier score measures the

##### mean squared difference between the predicted probability

##### assigned to the possible label and the actual label. The ECE

##### approximates the expectation between model confidence and

##### accuracy by binning the predictions into equally-spaced bins

##### and taking a weighted average of the bin’s accuracy and con-

##### fidence difference. For both Brier score and ECE, lower val-

##### ues indicate better calibrated models. A statistical significance

##### threshold (α) of 0.05 was used for all the reported tests.

## RESULTS

### Overall Summary

#### Between 2015 and 2017, 226 consecutive AF ablations

#### were done using a 64-pole basket catheter that recorded

#### simultaneous panoramic unipolar electrograms from the

#### left and the right atria. Of these, 161 had left atrial signals

#### recorded before any ablation. Five were excluded due to

#### poor signal quality, leaving 156 patients to be analyzed

#### for this study. Baseline characteristics of these patients

#### are shown in Table 1. PVI was done using radiofrequency

#### in 118 patients (76%), cryoballoon in 38 patients (24%).

#### Thirty-four patients (21.8%) were on an antiarrhythmic

#### drugs (AADs) at the time of follow up (10.2% on class

#### IC agents, 3.9% on class III agents [sotalol or dofetilide],

#### 8.3% on amiodarone and 1.9% on dronedarone). Addi-

#### tional ablation lesions beyond PVI and ablation of local-

#### ized sources are presented in Table 1.

### Catheter Ablation Outcomes

#### On follow-up at 1 year, 112 (72%) patients remained

#### free of AF. Patients with and without recurrence had

#### a similar age, body mass index and comorbidities

#### (Table 1). AAD use was not different among groups.

#### 28% of the patients had a prior history of AF ablation.

#### Presence of hyperlipidemia and diabetes correlated

#### with AF recurrence ( P =0.04) in univariate analysis.

#### Ablation of additional left atrial lines did not correlate

#### with AF ablation outcomes.

### Validation of Existing AF Ablation Outcome

### Prediction Scores: APPLE and CHA2DS2-VaSC

#### First, we validated 2 existing clinical feature-based pre-

#### diction scores, APPLE^3 and CHA2DS2-VaSC,^4 for 1-year

#### AF recurrence prediction using CatBoost.^17 Detailed for-

#### mulation of APPLE and CHA2DS2-VaSC scores can be

#### found in Supplemental Methods.

#### The CatBoost classifier achieved an AUROC of 0.

#### (SD=0.129) on APPLE scores and an AUROC of 0.

#### (SD=0.133) on CHA2DS2-VASc scores (Table 2, first

#### and second rows).

### ML-Based AF Recurrence Prediction From

### Clinical Features

#### Using clinical features, the CatBoost classifier achieved

#### an AUROC of 0.755 (SD=0.093; Table 2, third row), out-

#### performing the performance of the CatBoost classifier

#### trained on APPLE and CHA2DS2-VASc scores. This

#### performance improvement is expected given that mul-

#### tiple clinical features were used, whereas APPLE and

#### CHA2DS2-VASc scores only accounted for 5 and 7

#### clinical features, respectively.

#### Figure 2 shows the model interpretation of the clinical

#### features that contribute the most to AF recurrence pre-

#### diction in our clinical feature-based model, where the 5

#### most important features are left ventricular ejection frac-

#### tion, height, body mass index, weight, left atria volume

#### from CT, and left atria surface area; which have previously

#### been reported to correlate with development of incident

#### AF15,21 or poorer outcomes following AF ablation.16,

### ML-Based AF Recurrence Prediction From

### Electrogram or ECG

#### Using electrogram signals only, the CNN achieved

#### an AUROC of 0.731 (SD=0.105) for AF recurrence

###### Downloaded from http://ahajournals.org by on September 18, 2025


#### prediction (Table 2, 4th row); using ECG signals only,

#### the CNN achieved an AUROC of 0.767 (SD=0.122;

#### Table 2, 5th row), both of which outperform APPLE and

#### CHA2DS2-VASc scores.

#### In addition, we visualize examples of electrogram and

#### ECG learned by the CNNs using the Uniform Manifold

#### Approximation and Projection^23 (UMAP) dimensionality

#### reduction technique. As shown in Figure S2, the same

#### patient’s electrogram features are clustered together,

#### whereas different patients’ electrogram features are

#### further apart. Moreover, electrogram/ECG features

#### of patients with AF recurrence are further away from

###### Table 1. Baseline Characteristics of Population

```
All subjects
(n=156)
```
```
Free from AF
(n=112)
```
```
Recurrent AF
(n=44) P Value
```
```
Demographics
Age, y (mean±SD) 64.5±10.5 64.5±9.9 64.5±11.9 0.
Male sex, n (%) 115 (74%) 87 (78%) 28 (64%) 0.
Height (m, mean±SD) 1.77±0.1 1.77±0.1 1.77±0.1 0.
```
```
Weight (kg, mean±SD) 96.6±24.4 98.1±24.3 92.6±24.4 0.
BMI (kg/m^2 , mean±SD) 30.6±6.8 31.2±7.1 29.3±5.8 0.
Comorbidities
CAD, n (%) 30 (19%) 25 (22%) 5 (11%) 0.
```
```
CHF, n (%) 32 (21%) 25 (22%) 7 (16%) 0.
Hypertension, n (%) 104 (67%) 76 (68%) 28 (64%) 0.
Hyperlipidemia, n (%) 88 (56%) 69 (62%) 19 (43%) 0.
TIA or CVA, n (%) 13 (8%) 11 (10%) 2 (5%) 0.
```
```
Diabetes, n (%) 30 (19%) 26 (23%) 4 (9%) 0.
OSA, n (%) 59 (38%) 43 (38%) 16 (36%) 0.
CKD, n (%) 24 (15%) 17 (15%) 7 (16%) 0.
Prior AF ablation, n (%) 43 (28%) 26 (23%) 17 (39%) 0.
```
```
Type of AF 0.
Paroxysmal AF, n (%) 67 (43%) 47 (42%) 20 (46%)
Persistent AF, n (%) 66 (42%) 45 (40%) 21 (48%)
Long-standing persistent AF, n (%) 23 (15%) 20 (19%) 3 (7%)
```
```
AF ablation type* 0.
Left atrial linear ablation 38 (24%) 30 (34%) 8 (18%)
CTI 42 (27%) 35 (31%) 7 (16%)
Antiarrhythmic drug use 34 (22%) 26 (23%) 8 (18%) 0.
```
```
Values are n, mean±SD, or median (interquartile range). Categorical variables are compared using Fisher exact test; continuous
variables using the t test or Mann-Whitney U test if data are not normally distributed. AF indicates atrial fibrillation; BMI, body mass
index; CAD, coronary artery disease; CHF, congestive heart failure; CKD, chronic kidney disease; CTI, cavotricuspid isthmus ablation;
OSA, obstructive sleep apnea; and TIA, transient ischemic attack.
*In addition to pulmonary vein isolation and ablation of localized rotational and focal sources by FIRM mapping.
```
###### Table 2. Results of 1-Year AF Recurrence Prediction

```
AUROC Sensitivity Specificity Accuracy F1-score
APPLE score 0.644±0.129 0.915±0.138* 0.350±0.329 0.504±0.213 0.533±0.
CHA2DS2-VASc score 0.650±0.133 0.905±0.162 0.427±0.355 0.560±0.226 0.568±0.
Clinical Feature 0.755±0.093 0.875±0.137 0.680±0.198 0.728±0.121 0.656±0.
```
```
EGM 0.731±0.105 0.885±0.116 0.627±0.131 0.701±0.098 0.630±0.
ECG 0.767±0.122 0.812±0.176 0.770±0.183 0.781±0.112 0.682±0.
Fusion of EGM and clinical data 0.788±0.110 0.905±0.117 0.706±0.144 0.764±0.107 0.691±0.
Fusion of ECG and clinical data 0.836±0.063 0.865±0.112 0.812±0.124 0.827±0.070 0.747±0.
```
```
Fusion of EGM and ECG 0.833±0.084 0.915±0.138* 0.793±0.124 0.826±0.083 0.753±0.
Fusion of EGM, ECG and clinical feature 0.859±0.082* 0.870±0.200 0.867±0.121* 0.866±0.076* 0.784±0.106*
```
```
Values are mean±SD across 10-folds.
ECG indicates electrocardiogram; and EGM, electrogram.
*Best mean results for each metric.
```
###### Downloaded from http://ahajournals.org by on September 18, 2025


#### features of patients without AF recurrence, suggest-

#### ing that the CNNs are able to learn distinct patterns in

#### patients with different outcomes.

### ML-Based AF Recurrence Prediction From

### Fusion of Electrogram, ECG, and Clinical

### Features

#### Our final fusion model that combines electrogram, ECG,

#### and clinical features achieved an AUROC of 0.

#### (SD=0.082; Table 2, last row), outperforming the APPLE

#### scores, CHA2DS2-VASc scores, and ECG or electro-

#### gram signals alone, suggesting the effectiveness of our

#### fusion framework.

#### Figure 3 shows the ROC curves of the clinical fea-

#### ture-based models, the signal-based CNN models, and

#### the fusion model. At a low false positive rate, such as

#### 20% false positive rate, our fusion model had a true

#### positive rate (TPR) of 80%, which translates clinically to

#### missing 20% recurrent AF patients with 20% of the pre-

#### dicted recurrent AF being false positives. In contrast, the

#### CHA2DS2-VASc score-based classifier and the clinical

#### feature-based classifier only achieved a TPR of 40% and

#### 58%, respectively, which translates to missing 60% and

#### 42% recurrent AF patients, respectively, with the same

#### number of false positives.

#### Moreover, combining 2 modalities performed better

#### than single modalities (Table 2, 6th–8th rows), which

#### is intuitive given that 2 modalities encode additional

#### features than a single modality. Model performance in

#### various subgroups are provided in Supplemental Results

#### and Tables S4 through S6.

#### In addition to discriminative measures (eg, AUROC,

#### sensitivity, and specificity), we evaluate the calibration

#### of the models using Brier score^19 and expected cali-

#### bration error.^20 See Supplemental Results and Table

#### S7 for details.

## DISCUSSION

#### In this study, we developed a deep convolutional neural

#### network that encodes the spatiotemporal dependencies

#### in electrogram and ECG signals, as well as a multimodal

#### fusion framework that leverages clinical features, elec-

#### trogram, and ECG for predicting 1-year AF recurrence

#### after catheter ablation. Our study was based on a cohort

#### of 156 patients.

#### To our knowledge, compared with the existing AF

#### recurrence prediction scores to date, this provides the

#### highest performance in predicting which patients would

#### be free from AF 1 year following ablation.

#### Other studies evaluating prediction of AF ablation

#### outcomes using machine learning include Shade et al^9

#### that utilized ML and personalized computational model-

#### ing in 32 patients to predict AF recurrence following PVI

#### with either cryoballoon or radiofrequency approach. In

#### their machine learning model, their sources of informa-

#### tion (imaging, clinical data) were combined equally.^9 Late

###### Figure 2. Clinical feature-based

###### model interpretation.

###### Importance of clinical features in

###### predicting atrial fibrillation (AF) recurrence

###### using the CatBoost classifier (averaged

###### across 10 folds). The 5 most important

###### features are: left ventricular ejection

###### fraction (LVEF), height, body mass index

###### (BMI), weight, left atria volume from

###### computed tomography (CT), and left atria

###### surface area from CT.

###### Downloaded from http://ahajournals.org by on September 18, 2025


#### gadolinium enhanced magnetic resonance imaging scans

#### were used for imaging data. AUROC of 0.82 was reported

#### when clinical variables were included in the model. Firouz-

#### nia et al^10 extracted data from chest CT scans to establish

#### their association with likelihood of postablation AF recur-

#### rence in 203 patients using a random forest classifier.

#### Certain derived imaging features such as left atrial surface

#### area, volume, and sphericity index used in their study were

#### also included in our model as a part of clinical features.^10

#### PVI in this study was completed with either cryoballoon

#### or radiofrequency catheters. Moreover, posterior wall, sep-

#### tal, superior vena cava, and cavotricuspid isthmus ablation

#### were performed according to operator choice, although

#### further details of extra-PVI ablation were not discussed in

#### the study or included in the models.

#### In our study, all patients underwent PVI with cryobal-

#### loon or radiofrequency approach. Similar to Firouznia et

#### al,^10 patients undergoing various ablation strategies were

#### included, including ablation of localized sources detected

#### by FIRM mapping strategy in 100% of patients, left atrial

#### linear lesions in 24%, and cavotricuspid isthmus ablation

#### in 27% of patients. FIRM strategy was used in all patients

#### as it allowed simultaneous recording of unipolar signals

#### in the left atrium before any ablation in this cohort, which

#### was a prerequisite in our analysis. Our models were able to

#### predict long-term (1 year) freedom from arrhythmias inde-

#### pendent of the ablation strategy. Clinical benefit of lesions

#### beyond PVI in patients with persistent AF has been a sub-

#### ject of debate, with multiple studies showing no additional

#### benefit of extra PV lesions in long term freedom from

#### A F,24,25 with some demonstrating incremental benefit,26,

#### and larger multicenter studies underway to evaluate this

#### further.^28 Furthermore, incorporation of intracardiac elec-

#### trograms indeed improved prediction of AF ablation out-

#### comes, suggesting that an AF mechanism might be at

#### play that could be delineated further by feature interpreta-

#### tion of these signals. Given the wide variety of ablation

#### approaches used in the training and testing cohorts for

#### our machine learning model, and limited representation of

#### subgroups such as women, generalizability of our findings

#### to the broader population could be limited.

### Limitations

#### This study was performed at a single center, involves a

#### small cohort with underrepresentation of women, and

#### results have not been validated externally. Heterogene-

#### ity in ablation approaches may limit generalizability of

#### the findings to specific ablation strategies. Despite this

#### limitation, all the patients underwent PVI, and evidence

#### of benefit of further ablation beyond PVI, including lin-

#### ear ablation and ablation of sites of organized rotational

#### or focal activation, has not been proven consistently in

#### multicenter randomized studies.24,29,30 All patients in this

#### study underwent FIRM mapping and ablation that formed

#### the basis of the unipolar EGMs used in the model. The

#### necessity of the use of FIRM mapping is a limitation to

#### this study, as this is not a widely used catheter or map-

#### ping strategy in the community.

#### Freedom from AF appears higher for a mixed cohort

#### of patients but is consistent with other studies that used

#### intermittent monitoring rather than implanted loop record-

#### ers. Intermittent monitoring of AF recurrence with 12-lead

#### ECGs and 14-day event monitors likely underrepresents

#### true AF recurrence, which could affect the accuracy of our

#### predictive model. The retrospective nature of the data lim-

#### ited strict guidelines over AAD use in follow up, that is, for

#### certain patients, preprocedure AADs were continued pos-

#### tablation due to patient or provider preference regardless

#### of procedure outcome. Twenty-eight percent of patients

#### had prior AF ablation, which may have impacted intra-

#### cardiac signal characteristics. Twelve-lead ECGs in sinus

#### rhythm before ablation were not available in all patients.

#### When a patient’s 12-lead ECGs in sinus rhythm before

#### ablation was not available, a 12-lead ECG in sinus rhythm

#### immediately after ablation was used for analysis, which

#### could result in bias in analyses. While we show that when

#### evaluating the trained models on patients whose preabla-

#### tion 12-lead ECGs are available, the model performance

#### did not differ significantly from our original analysis (ie,

#### post-ablation 12-lead ECGs are used for patients whose

#### preablation ECGs are not available), we did not re-train the

#### models on preablation ECGs only due to the limited size of

#### our cohort (n=107 patients with preablation ECGs). Major-

#### ity of these patients had a 12-lead ECG in sinus rhythm

#### before ablation that was not performed at our center in

###### Figure 3. Receiver operating characteristics (ROC) curves of

###### the clinical feature-based models, signal-based models, and

###### the fusion model.

###### The x -axis shows the false positive rate averaged across 10 folds

###### for each model, and the y -axis shows the true positive rate averaged

###### across 10 folds for each model. AUROC indicates area under the

###### receiver operating characteristics curve.

###### Downloaded from http://ahajournals.org by on September 18, 2025


#### an electronic format that could be exported for analysis,

#### due to the tertiary referral center status where the study

#### was conducted. Some of the data that were used in the

#### models to predict ablation success, including intracardiac

#### signals, are obtained at the time of ablation, and may not

#### help in patient selection for ablation procedure, but can

#### rather guide medical management and expectations fol-

#### lowing the procedure. Furthermore, while we show that

#### most of the trained models perform similarly on patient

#### subgroups (patients with paroxysmal versus nonparoxys-

#### mal AF; patients with cryoablation versus radiofrequency

#### ablation), future study with a larger cohort that trains mod-

#### els on these subgroups independently is needed to fur-

#### ther compare these subgroups. Last, while we show that

#### our CNNs and fusion model are better calibrated than the

#### existing APPLE and CHASDS2-VASc scores (Table S7),

#### the Brier scores and expected calibration errors are still

#### relatively high; advanced calibration techniques^31 for deep

#### neural networks need to be incorporated in the future to

#### produce better calibrated models.

### Conclusions

#### Our machine learning approach provides an automatic

#### technique to predict freedom from atrial arrhythmias in

#### patients undergoing AF ablation, outperforming tradi-

#### tional scoring systems. Larger datasets are needed in

#### the future to train and validate this approach even fur-

#### ther to help develop personalized ablation strategies for

#### patients with AF.

#### ARTICLE INFORMATION

```
Received January 12, 2022; accepted June 29, 2022.
```
##### Affiliations

```
Department of Electrical Engineering (S.T.); Department of Medicine, Division of
Cardiovascular Medicine (R.K., M.I.A., M.F., A.J.R., M.R.B., P.C., P.J.W., S.M.N., T.B);
Department of Radiology and Biomedical Data Science (D.L.R.), Stanford Univer-
sity, CA. University College London, Centre for Advanced Research Computing,
United Kingdom (O.R.). CoMMLab, Universitat de Valencia, VA, Spain (M.R.B.).
```
##### Sources of Funding

```
This work was funded by National Institutes of Health (K23 HL145017) grant
to Dr Baykaner.
```
##### Disclosures

```
M.I. Alhusseini reports intellectual property rights from Stanford University. Dr
Fazal reports no disclosures. Dr Rubin reports grants from NIH (1U01CA190214,
1U01CA187947, U01CA242879, U24CA226110) and consulting fees from
Roche-Genentech, Dr Narayan reports research grants from NIH (HL70529,
HL103800, HL83359), consulting from LifeSignals.ai Inc, TDK Inc., Up to Date,
Abbott Laboratories, and American College of Cardiology Foundation (all mod-
est); Intellectual Property Rights from University of California Regents and Stan-
ford University. Dr Baykaner reports funding from NIH (K23 HL145017) and
consulting fees from Medtronic, BIOTRONIK and PaceMate.
```
##### Supplemental Material

```
Supplemental Methods
Supplemental Results
Tables S1–S
Figures S1 and S
References32–
```
#### REFERENCES

1. Packer DL, Mark DB, Robb RA, Monahan KH, Bahnson TD, Poole JE,
    Noseworthy PA, Rosenberg YD, Jeffries N, Mitchell LB, et al; CABANA
    Investigators. Effect of catheter ablation vs antiarrhythmic drug ther-
    apy on mortality, stroke, bleeding, and cardiac arrest among patients
    with atrial fibrillation: the CABANA Randomized Clinical Trial. _JAMA_.
    2019;321:1261–1274. doi: 10.1001/jama.2019.
2. Marrouche NF, Brachmann J, Andresen D, Siebels J, Boersma L,
    Jordaens L, Merkely B, Pokushalov E, Sanders P, Proff J, et al; CASTLE-
    AF Investigators. Catheter ablation for atrial fibrillation with heart failure. _N_
    _Engl J Med_. 2018;378:417–427. doi: 10.1056/NEJMoa
3. Kornej J, Hindricks G, Arya A, Sommer P, Husser D, Bollmann A. The
APPLE score - a novel score for the prediction of rhythm outcomes after
repeat catheter ablation of atrial fibrillation. _PLoS One_. 2017;12:e0169933.
doi: 10.1371/journal.pone.
4. Jacobs V, May HT, Bair TL, Crandall BG, Cutler M, Day JD, Weiss JP,
Osborn JS, Muhlestein JB, Anderson JL, et al. The impact of risk score
(CHADS2 versus CHA2DS2-VASc) on long-term outcomes after
atrial fibrillation ablation. _Heart Rhythm_. 2015;12:681–686. doi:
10.1016/j.hrthm.2014.12.
5. Kosich F, Schumacher K, Potpara T, Lip GY, Hindricks G, Kornej J. Clinical
scores used for the prediction of negative events in patients undergoing
catheter ablation for atrial fibrillation. _Clin Cardiol_. 2019;42:320–329. doi:
10.1002/clc.
6. Rogers AJ, Selvalingam A, Alhusseini MI, Krummen DE, Corrado C,
Abuzaid F, Baykaner T, Meyer C, Clopton P, Giles W, et al. Machine learned
cellular phenotypes in cardiomyopathy predict sudden death. _Circ Res_.
2021;128:172–184. doi: 10.1161/CIRCRESAHA.120.
7. Bhalodia R, Goparaju A, Sodergren T, Morris A, Kholmovski E, Marrouche N,
Cates J, Whitaker R, Elhabian S. Deep Learning for End-to-End Atrial Fibril-
lation Recurrence Estimation. In: _2018 Computing in Cardiology Conference
(CinC)_. IEEE; 2018. p. 1–4. DOI: 10.22489/CinC.2018.
8. Budzianowski J, Hiczkiewicz J, Burchardt P, Pieszko K, Rzeźniczak J,
Budzianowski P, Korybalska K. Predictors of atrial fibrillation early recur-
rence following cryoballoon ablation of pulmonary veins using statistical
assessment and machine learning algorithms. _Heart Vessels_. 2019;34:352–
359. doi: 10.1007/s00380-018-1244-z
9. Shade JK, Ali RL, Basile D, Popescu D, Akhtar T, Marine JE, Spragg DD,
Calkins H, Trayanova NA. Preprocedure application of machine learning and
mechanistic simulations predicts likelihood of paroxysmal atrial fibrillation
recurrence following pulmonary vein isolation. _Circ Arrhythm Electrophysiol_.
2020;13:e008213. doi: 10.1161/CIRCEP.119.
10. Firouznia M, Feeny AK, LaBarbera MA, McHale M, Cantlay C, Kalfas N,
Schoenhagen P, Saliba W, Tchou P, Barnard J, et al. Machine learning-
derived fractal features of shape and texture of the left atrium and pulmo-
nary veins from cardiac computed tomography scans are associated with
risk of recurrence of atrial fibrillation postablation. _Circ Arrhythm Electro-
physiol_. 2021;14:e009265. doi: 10.1161/CIRCEP.120.
11. LeCun Y, Bengio Y, Hinton G. Deep learning. _Nature_. 2015;521:436–444.
doi: 10.1038/nature
12. Esteva A, Robicquet A, Ramsundar B, Kuleshov V, DePristo M, Chou K,
Cui C, Corrado G, Thrun S, Dean J. A guide to deep learning in healthcare.
_Nat Med_. 2019;25:24–29. doi: 10.1038/s41591-018-0316-z
13. Firouznia M, Feeny AK, LaBarbera MA, McHale M, Cantlay C, Kalfas N,
Schoenhagen P, Saliba W, Tchou P, Barnard J, et al. Machine learning–
derived fractal features of shape and texture of the left atrium and
pulmonary veins from cardiac computed tomography scans are asso-
ciated with risk of recurrence of atrial fibrillation postablation. _Cir-
culation: Arrhythmia and Electrophysiology_. 2021;14:e009265. doi:
10.1161/CIRCEP.120.
14. Attia ZI, Noseworthy PA, Lopez-Jimenez F, Asirvatham SJ, Deshmukh AJ,
Gersh BJ, Carter RE, Yao X, Rabinstein AA, Erickson BJ, et al. An artificial
intelligence-enabled ECG algorithm for the identification of patients with atrial
fibrillation during sinus rhythm: a retrospective analysis of outcome prediction.
_Lancet_. 2019;394:861–867. doi: 10.1016/S0140-6736(19)31721-
15. Lavie CJ, Pandey A, Lau DH, Alpert MA, Sanders P. Obesity and atrial
fibrillation prevalence, pathogenesis, and prognosis: effects of weight loss
and exercise. _J Am Coll Cardiol_. 2017;70:2022–2035. doi: 10.1016/j.
jacc.2017.09.
16. Khaykin Y, Oosthuizen R, Zarnett L, Essebag V, Parkash R, Seabrook C,
Beardsall M, Tsang B, Wulffhart Z, Verma A. Clinical predictors of arrhythmia
recurrences following pulmonary vein antrum isolation for atrial fibrillation:
predicting arrhythmia recurrence post-PVAI. _J Cardiovasc Electrophysiol_.
2011;22:1206–1214. doi: 10.1111/j.1540-8167.2011.02108.x

###### Downloaded from http://ahajournals.org by on September 18, 2025


17. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost:
    unbiased boosting with categorical features. In: _Advances in Neural Informa-_
    _tion Processing Systems_. 2018.
18. Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. In: _3rd Inter-_
    _national Conference on Learning Representations_ , ICLR 2015, San Diego,
    CA, USA, May 7-9, 2015, Conference Track Proceedings. 2015.
19. Brier GW. Verification of forecasts expressed in terms of probability. _Mon_
    _Weather Rev_. 1950;78:1–3. Accessable at: https://books.google.com/book
    s?hl=en&lr=&id=jnbpAAAAMAAJ&oi=fnd&pg=PA11-IA6&ots=0X13-JoHz
    M&sig=66qS3nq3yYAE0mAcaPH2bVkGCHE#v=onepage&q&f=false
20. Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities
using bayesian binning. _Proc Conf AAAI Artif Intell_. 2015;2015:2901–2907.
21. Santhanakrishnan R, Wang N, Larson MG, Magnani JW, McManus DD,
Lubitz SA, Ellinor PT, Cheng S, Vasan RS, Lee DS, et al. Atrial fibrillation
begets heart failure and vice versa: temporal associations and differences
in preserved versus reduced ejection fraction. _Circulation_. 2016;133:484–
492. doi: 10.1161/CIRCULATIONAHA.115.
22. Pathak RK, Middeldorp ME, Lau DH, Mehta AB, Mahajan R, Twomey D,
Alasady M, Hanley L, Antic NA, McEvoy RD, et al. Aggressive risk factor
reduction study for atrial fibrillation and implications for the outcome of
ablation: the ARREST-AF cohort study. _J Am Coll Cardiol_. 2014;64:2222–
2231. doi: 10.1016/j.jacc.2014.09.
23. McInnes L, Healy J, Melville J. UMAP: uniform manifold approximation and
projection for dimension reduction. _arXiv [stat_. _ML]_. 2018. doi: https://doi.
org/10.48550/arXiv.1802.
24. Verma A, Jiang CY, Betts TR, Chen J, Deisenhofer I, Mantovan R, Macle L,
Morillo CA, Haverkamp W, Weerasooriya R, et al; STAR AF II Investiga-
tors. Approaches to catheter ablation for persistent atrial fibrillation. _N
Engl J Med_. 2015;372:1812–1822. doi: 10.1056/NEJMoa
25. Brachmann J, Hummel JD, Wilber DJ, Sarver AE, Rapkin J, Shpun S,
Szili-Torok T. Prospective randomized comparison of rotor ablation vs con-
ventional ablation for treatment of persistent atrial fibrillation—The REAF-
FIRM trial. _Heart Rhythm_. 2019;16:963–965.
26. Clarke JD, Piccini JP, Friedman DJ. The role of posterior wall isolation in
catheter ablation of persistent atrial fibrillation. _J Cardiovasc Electrophysiol_.
2021;32:2567–2576. doi: 10.1111/jce.
27. Narayan SM, Baykaner T, Clopton P, Schricker A, Lalani GG, Krummen DE,
Shivkumar K, Miller JM. Ablation of rotor and focal sources reduces late
recurrence of atrial fibrillation compared with trigger ablation alone: extended
follow-up of the CONFIRM trial (Conventional Ablation for Atrial Fibrillation
With or Without Focal Impulse and Rotor Modulation). _J Am Coll Cardiol_.
2014;63:1761–1768. doi: 10.1016/j.jacc.2014.02.
28. Terricabras M, Piccini JP, Verma A. Ablation of persistent atrial fibrillation:
Challenges and solutions. _J Cardiovasc Electrophysiol_. 2020;31:1809–
1821. doi: 10.1111/jce.
29. Thiyagarajah A, Kadhim K, Lau DH, Emami M, Linz D, Khokhar
K, Munawar DA, Mishima R, Malik V, O’Shea C, et al. Feasibility, safety,
and efficacy of posterior wall isolation during atrial fibrillation ablation:
a systematic review and meta-analysis. _Circ Arrhythm Electrophysiol_.
2019;12:e007005. doi: 10.1161/CIRCEP.118.
30. Kirzner JM, Raelson CA, Liu CF, Thomas G, Ip JE, Lerman BB,
Markowitz SM, Cheung JW. Effects of focal impulse and rotor modulation-
guided ablation on atrial arrhythmia termination and inducibility: Impact
on outcomes after treatment of persistent atrial fibrillation. _J Cardiovasc
Electrophysiol_. 2019;30:2773–2781. doi: 10.1111/jce.
31. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural
Networks. In: Precup D, Teh YW, editors. _Proceedings of the 34th International
Conference on Machine Learning_. PMLR; 2017. p. 1321–1330.
32. Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network
Training by Reducing Internal Covariate Shift. In: Bach F, Blei D, editors.
_Proceedings of the 32nd International Conference on Machine Learning_.
Lille, France: PMLR; 2015. p. 448–456.
33. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition.
In: 2016 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR). _IEEE_ ; 2016: pp. 770-778.
34. Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R. Drop-
out: a simple way to prevent neural networks from overfitting. _J Mach Learn
Res_. 2014;15:1929–1958. doi: 10.1109/TCYB.2020.
35. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas
under two or more correlated receiver operating characteristic curves:
a nonparametric approach. _Biometrics_. 1988;44:837–845. doi:
10.2307/

###### Downloaded from http://ahajournals.org by on September 18, 2025




# SUPPLEMENTAL MATERIAL


**Supplemental Methods**

_I. Convolutional neural network_

In the convolutional neural network (CNN), each bottleneck block **(Supplemental Figure I**

**panel A)** in the time-dimension convolutional layersconsists of two branches: (a) two

convolutional layers, each of which consists of a batch normalization^32 , a ReLU non-linear

activation, and a one-dimensional (1D) convolution operating on the time dimension, (b) a

shortcut connection with a 1x1 convolution followed by a max-pooling layer, where the shortcut

connection allows information to flow easily from previous layers and enables easier

optimization of deep neural networks^33. The outputsfrom these two branches were added

together to form the output of a bottleneck block. The time-dimension convolutional layers

consist of several stacked bottleneck blocks, followed by an average-pooling layer. Next, the

channel-dimension convolutional layer consists of a 1D convolution operating on the channel

dimension, a batch normalization, and a ReLU non-linearity. Finally, a fully connected layer with

sigmoid activation is used to obtain the predicted probability of AF recurrence for the input

signal window. To mitigate overfitting problems, dropout layers^34 were inserted between the

bottleneck blocks as well as before the last fully connected layer. The probabilities from all 5-sec

windows of the same patient were averaged to obtain the final predicted probability of AF

recurrence for each patient. For EGM-based CNN, model hyperparameters (i.e., kernel size,

output channel size, number of bottleneck blocks, dropout probability) were configured to be the

same as that in Attia et al.^14 ; for ECG-based CNN,the number of bottleneck blocks were reduced

to 6 (versus 9) because of the reduced number of 5-sec ECG windows for model training. Details

of the CNN architecture can be found in **SupplementalFigure I panels B–C**.


_II. APPLE and CHA2DS2-VASc scores_

APPLE scores^3 are calculated as follows: (a) add onepoint for age > 65 years, (b) add one point

for persistent or longs-tanding persistent atrial fibrillation (AF), (c) add one point for chronic

kidney disease (CKD), (d) add one point if left atria (LA) diameter ≥ 43mm, (e) add one point if

left ventricular ejection fraction (LVEF) < 50%.

CHA2DS2-VASc scores^4 are calculated as follows: (a)add one point for congestive heart failure

(CHF), (b) add one point for hypertension, (c) add two points for age ≥ 75 years, (d) add one

point for diabetes mellitus (DM), (e) add two points for TIA or CVA, (f) add one point for

coronary artery disease (CAD) or peripheral artery disease, (g) add one point for age between 65

and 74 years, (h) add one point for female.

_III. QRS subtraction_

QRS subtraction for EGM was performed as follows. First, the Q, R, and S points in the three

surface ECG channels (I, II, III) were identified. Second, the pythagorean distance from these

QRS points was calculated. Third, the average Q-R time distance and the average R-S time

distance were calculated, and a buffer of -40 millisecond from the Q point and a buffer of +

millisecond from the S point were used to find the width of the QRS segment for all the three

ECG channels. Fourth, the average QRS of each basket channel was calculated as follows: (a)

find the corresponding QRS signal in the basket channel using the QRS time points obtained in

prior steps, (b) remove baseline offset from each QRS by subtracting the first point from the

signal, and (c) calculate the average QRS in this channel. Fifth, a bandpass filter of 0.5–100 Hz

was applied to the signal. Sixth, the QRS segment was removed from the basket channels by


subtracting the average QRS of the basket signal for each QRS time point. Lastly, all EGM

signals were resampled to 200 Hz.

_IV. Model hyperparameter tuning_

Hyperparameters for CNNs were configured to be the same as that in Attia et al.^14 (except for

reducing the number of bottleneck blocks in ECG-based CNN; see section I in Supplemental

Methods) and no hyperparameter tuning was involved. Hyperparameters for the CatBoost^17

classifier were tuned using a grid search using 5-fold cross-validation on the training folds,

including (a) number of iterations within {1000, 2000, 5000}, (b) learning rate within {0.001,

0.01, 0.1}, (c) depth within {2, 5, 10}, (d) L2 leaf regularization within {1, 3, 5}.


**Supplemental Results**

_I. Effects of missing value imputation_

To investigate the effect of our imputation method (i.e., imputing missing clinical values using

the most frequent values) on model performance, we re-trained models involving clinical

features using two additional imputation methods: (a) imputing missing values using the mean

values and (b) imputing missing values using the median values. **Supplemental Table II** shows

the model performance of these two imputation methods and their comparisons to the original

imputation method (most frequent). There is no statistically significant difference among these

three imputation methods. Moreover, **Supplemental TableIII** shows performance of the models

when trained only on patients without any missing clinical features (n = 114). Similarly, there is

no statistically significant difference between performance on the entire cohort and performance

on the subset without missing values. These results suggest that imputing missing values with the

most frequent ones does not introduce bias in the input data in our study.

_II. Model performance on patients with paroxysmal AF versus patients with non-paroxysmal AF_

Here, we evaluate model performance on two subgroups of patients: patients with paroxysmal

AF (n = 66) and patients with non-paroxysmal AF (n = 90) (i.e., persistent and long-standing

persistent). As shown in **Supplemental Table IV** , fusionmodels result in similar prediction

performance between the entire cohort and these two subgroups of patients (no statistically

significant difference), and outperform clinical feature-based models and EGM/ECG-based

models.


_III. Model performance on patients with cryoablation versus patients with radiofrequency_

_ablation_

Furthermore, we evaluate model performance on patients with cryoablation (n = 38) and patients

with radiofrequency ablation (n = 111). As shown in **Supplemental Table V** , fusion models

result in similar prediction performance between the entire cohort and these two subgroups, and

outperform clinical feature-based models and EGM/ECG-based models. We note that model

performance on patients with cryoablation has high variance, which is likely due to the small

number of patients who received cryoablation in our cohort (n = 38 in total; 4-5 patients in each

test fold on average).

_IV. Effects of the use of post-ablation ECGs_

Among 156 patients in our cohort, 49 patients do not have pre-ablation ECGs. We used their

ECGs in sinus rhythm immediately after ablation for these 49 patients. To examine if the use of

post-ablation ECGs affects model performance, we evaluate the models using ECGs on the

subset of patients whose pre-ablation ECGs are available (n = 107). As shown in **Supplemental**

**Table VI** , models perform comparably between the entirecohort and patients with pre-ablation

ECGs (no statistically significant difference).


_V. Evaluation of calibration of models_

In addition to discriminative measures (e.g., AUROC, sensitivity, and specificity), we evaluate

the calibration of the models using Brier score^19 and expected calibration error (ECE)^20. As

shown in Supplemental Table VII, we observe that our EGM and ECG-based CNNs and fusion

models have lower Brier scores and ECEs than APPLE score, CHASDS2-VASc score, and the

clinical feature-based classifier. This suggests that our models not only provide higher prediction

performance, but are also better calibrated than the existing clinical scores.


## Supplemental Tables

**Supplemental Table I**. List of clinical features usedin this study and the number of missing

values in each feature.


- Type of AF Clinical Features No. Missing Values
- Age
- Sex
- Height
- Weight
- Body Mass Index (BMI)
- Prior AF Ablation
- Congestive Heart Failure (CHF)
- Left Ventricular Ejection Fraction (LVEF)
- Hypertension (HTN)
- Hyperlipidemia (HLD)
- TIA/CVA
- Coronary Artery Disease (CAD)
- History of Myocardial Infarction (MI)
- Percutaneous Coronary Intervention (PCI)
- Valvular Disease
- History of Ventricular Tachycardia (VT)
- Congenital Heart Disease
- Asthma
- Obstructive Sleep Apnea (OSA)
- Diabetes Mellitus (DM)
- Chronic Kidney Disease (CKD)
- Left Atrial Surface Area From CT
- Left Atrial Volume From CT
- Left Atrial Volume From ECHO
- Left Atrial Diameter > 42mm
- Left Atrial Sphericity Index


**Supplemental Table II**. Comparison of different imputationmethods on models using clinical

features. Values are mean ± standard deviation across 10-folds.

*p-values compare between most frequent imputationand mean/median imputation. p-values are

computed using DeLong’s nonparametric test^35 , wheretest fold patients’ predictions in 10 folds

are aggregated together prior to the test.

```
Most Frequent
Imputation
(Original) Mean Imputation Median Imputation
AUROC AUROC p-value* AUROC p-value*
APPLE Score 0.644 ± 0.129 0.644 ± 0.129 1.000 0.644 ± 0.129 1.
```
```
CHA2DS2-VASc
Score 0.650 ± 0.133 0.650 ± 0.133 1.000 0.650 ± 0.133 1.
```
```
Clinical Feature 0.755 ± 0.093 0.743 ± 0.108 0.385 0.770 ± 0.085 0.
Fusion of EGM
& Clinical Data 0.788 ± 0.110 0.781 ± 0.125 0.743 0.786 ± 0.113 0.
Fusion of ECG
& Clinical Data 0.836 ± 0.063 0.837 ± 0.050 0.995 0.840 ± 0.053 0.
Fusion of EGM,
ECG & Clinical
Feature 0.859 ± 0.082 0.839 ± 0.093 0.745 0.854 ± 0.099 0.
```

**Supplemental Table III.** Performance of models usingclinical features on patients without any

missing values. Values are mean ± standard deviation across 10-folds.

*p-values are computed using DeLong’s nonparametrictest (^35) , where test patients’ predictions in
10 folds are aggregated together prior to the test.
**All patients (n = 156)
AUROC
Patients without any
missing values (n = 114)
AUROC
p-value*
APPLE Score** 0.644 ± 0.129 0.654 ± 0.168 0.
**CHA2DS2-VASc Score** 0.650 ± 0.133 0.691 ± 0.132 0.
**Clinical Feature** 0.755 ± 0.093 0.733 ± 0.170 0.
**Fusion of EGM &
Clinical Data** 0.788 ± 0.110 0.777 ± 0.122 0.
**Fusion of ECG &
Clinical Data** 0.836 ± 0.063 0.834 ± 0.116 0.
**Fusion of EGM, ECG &
Clinical Feature** 0.859 ± 0.082 0.849 ± 0.100 0.


**Supplemental Table IV**. Model performance on patientswith paroxysmal AF versus patients

with non-paroxysmal AF (persistent and long-standing persistent). Values are mean ± standard

deviation across 10-folds. Best mean results for each group of patients are highlighted in bold.

*p-values compare between all patients and patientswith paroxysmal/non-paroxysmal AF.

p-values are computed using DeLong’s nonparametric test^35 , where test patients’ predictions in

10 folds are aggregated together prior to the test.

```
All Patients
(n = 156)
```
```
Patients with
Paroxysmal AF
(n = 66)
```
```
Patients with
Non-Paroxysmal AF
(n = 90)
```
```
AUROC AUROC p-value* AUROC p-value*
APPLE Score 0.644 ± 0.129 0.584 ± 0.224 0.603 0.630 ± 0.117 0.
CHA2DS2-VAS
c Score 0.650 ± 0.133 0.634 ± 0.191 0.794 0.654 ± 0.206 0.
Clinical Feature 0.755 ± 0.093 0.811 ± 0.158 0.864 0.673 ± 0.166 0.
```
```
EGM 0.731 ± 0.105 0.641 ± 0.316 < 0.001 0.680 ± 0.248 0.
ECG 0.767 ± 0.122 0.859 ± 0.128 0.335 0.575 ± 0.196 0.
```
```
Fusion of EGM
& Clinical Data 0.788 ± 0.110 0.774 ± 0.219 0.635 0.753 ± 0.234 0.
```
```
Fusion of ECG
& Clinical Data 0.836 ± 0.063 0.868 ± 0.164 0.880 0.780 ± 0.134 0.
```
```
Fusion of EGM
& ECG 0.833 ± 0.084 0.712 ± 0.242 0.781 0.895 ± 0.113 0.
```
```
Fusion of EGM,
ECG & Clinical
Feature 0.859 ± 0.082 0.909 ± 0.147 0.431 0.871 ± 0.147 0.
```

**Supplemental Table V**. Model performance on patientswith cryoablation versus patients with

radiofrequency ablation. Values are mean ± standard deviation across 10-folds. Best mean results

for each group of patients are highlighted in bold.

*p-values compare between all patients and patientswith cryoablation/radiofrequency ablation.

p-values are computed using DeLong’s nonparametric test^35 , where test patients’ predictions in

10 folds are aggregated together prior to the test.

```
All Patients
```
```
Patients with
Cryoablation
(n = 38)
```
```
Patients with
Radiofrequency Ablation
(n = 111)
AUROC AUROC p-value* AUROC p-value*
```
```
APPLE Score 0.644 ± 0.129 0.583 ± 0.167 0.677 0.632 ± 0.129 0.
CHA2DS2-VASc
Score 0.650 ± 0.133 0.724 ± 0.334 0.388 0.644 ± 0.177 0.
```
```
Clinical Feature 0.755 ± 0.093 0.693 ± 0.215 0.960 0.787 ± 0.098 0.
```
```
EGM 0.731 ± 0.105 0.438 ± 0.399 < 0.001 0.729 ± 0.120 < 0.
```
```
ECG 0.767 ± 0.122 0.469 ± 0.328 0.983 0.800 ± 0.117 0.
```
```
Fusion of EGM &
Clinical Data 0.788 ± 0.110 0.854 ± 0.256 0.880 0.751 ± 0.167 0.
```
```
Fusion of ECG &
Clinical Data 0.836 ± 0.063 0.760 ± 0.307 0.455 0.896 ± 0.084 0.
```
```
Fusion of EGM &
ECG 0.833 ± 0.084 0.906 ± 0.174 0.390 0.839 ± 0.108 0.
```
```
Fusion of EGM,
ECG & Clinical
Feature 0.859 ± 0.082 1.000 ± 0.000 0.671 0.861 ± 0.090 0.
```

**Supplemental Table VI.** Performance of models usingECG data on patients whose

prior-ablation ECGs are available. Values are mean ± standard deviation across 10-folds.

*p-values are computed using DeLong’s nonparametrictest (^35) , where test patients’ predictions in
10 folds are aggregated together prior to the test.
**All Patients (n = 156)
AUROC
Patients with
Prior-Ablation ECG
(n = 107)
AUROC
p-value***

### ECG 0.767 ± 0.122 0.677 ± 0.201 0.

```
Fusion of ECG &
Clinical Data 0.836 ± 0.063 0.810 ± 0.107 0.
```
```
Fusion of EGM & ECG 0.833 ± 0.084 0.836 ± 0.117 0.
Fusion of EGM, ECG &
Clinical Data 0.859 ± 0.082 0.889 ± 0.125 0.
```

**Supplemental Table VII. Brier score and expected calibration error (ECE) of baseline and**

**our proposed models.** Smaller Brier score and ECE indicatebetter calibrated models. Values are

mean± standard deviation across 10-folds.

```
Brier Score Expected
Calibration Error
(ECE)
```
```
APPLE Score
```
### 0.242 ± 0.013 0.199 ± 0.

```
CHA2DS2-VASc Score
```
### 0.236 ± 0.022 0.193 ± 0.

```
Clinical Feature
```
### 0.231 ± 0.019 0.188 ± 0.

### EGM

### 0.198 ± 0.026 0.147± 0.

### ECG

### 0.199 ± 0.023 0.132 ± 0.

```
Fusion of EGM &
Clinical Data
```
### 0.210 ± 0.029 0.200 ± 0.

```
Fusion of ECG &
Clinical Data
```
### 0.201 ± 0.051 0.169 ± 0.

```
Fusion of EGM & ECG 0.197 ± 0.037 0.186 ± 0.
```
```
Fusion of EGM, ECG &
Clinical Feature
```
### 0.206 ± 0.039 0.190 ± 0.


**Supplemental Figures and Figure Legends**


**Supplemental Figure I. (A) Illustration of a bottleneck block**^33. There are two branches. The

first branch consists of two layers of (1) batch normalization^32 ,(2) ReLU nonlinearity activation,

and (3) 1-dimensional convolution operated on the time dimension. The second branch consists

of a shortcut connection with a 1x1 convolution and a max-pooling layer. The output from these

two branches were added together to produce the output of the bottleneck block. The input has

shape L x C x ch1, where L indicates the time dimension, C indicates the EGM/ECG

lead/channel dimension, and ch1 indicates the input channel dimension. The resulting output has

shape L/2 x C x ch2, where ch2 is the output channel dimension. **(B)–(C) CNN architecture for**

**EGM and ECG**. The CNN consists of several bottleneckblocks operating on the time

dimension with dropout layers, followed by an average-pooling layer and a final convolutional

layer operating on the channel dimension. Finally, a fully connected (FC) layer with sigmoid

activation is applied to produce the predicted probability of AF recurrence. Moreover, the input

EGM or ECG matrices were padded to length 1024 in the time dimension with zeros to ensure

that the max-pooling layers in the bottleneck blocks resulted in integer output lengths. We note

that there are fewer ECG data (i.e., 5-sec windows) compared to EGM data, and thus the CNN

architecture for ECG consists of fewer bottleneck blocks to reduce the number of model

parameters.


**Supplemental Figure II.** Visualization of EGM (left)and ECG (right) features learned by the

convolutional neural networks (CNN) using Uniform Manifold Approximation and Projection^23

(UMAP) in the test set of the median-performing folds (n=15 patients for EGM-based CNN and

n=16 patients for ECG-based CNN). Specifically, the 128-dimensional EGM/ECG features are

reduced to 2 dimensions using the UMAP dimensionality reduction algorithm and visualized

here. Each dot represents features for one 5-second EGM/ECG window; blue dots represent

patients without AF recurrence, while brown dots represent patients with AF recurrence. On the

left panel, each cluster corresponds to one patient, suggesting that the same patient’s features are

more similar. On both panels, blue dots are further away from brown dots, suggesting that the

models learned distinct features in patients with different outcomes.


