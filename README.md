# awesome-auto-annotation

When training a new AI pipeline, the annotation process can be painful and quite costly. In most of the project, 
it is one of the main bottleneck in the release of impactful AI pipelines. Then, leveraging existing AI pipelines to 
automatically generate new annotations is a way to acccelerate this process.

This repository is aiming to provide a curated list of existing solutions allowing to automatically generate annotations. 
The information is split by data type (image, text, video, sound, ...) and each of this section is giving:
* list of public repositories
* list of private tools
* list of scientific publications

# image-auto-annotation

## public repositories

### backend only

* [AlvaroCavalcante/auto_annotate](https://github.com/AlvaroCavalcante/auto_annotate): Locally annotate from a pretrained 
    object detection model after setting up a threshold
* [Nvidia Tao Auto Label](https://docs.nvidia.com/tao/tao-toolkit/text/data_services/auto-label.html): run GroundingDINO 
    to annotate images (local or cloud)
* [mailcorahul/auto_labeler](https://github.com/mailcorahul/auto_labeler): CLI tools allowing to leverage different
    foundation models to generate annotations on local data
* [autodistill/autodistill](https://github.com/autodistill/autodistill): Locally annotate from a foundation model and train
    a specific model on top of it.
* [ultralytics/ultralytics](https://docs.ultralytics.com/models/sam-2/#sam-2-comparison-vs-yolo): Auto annotate from Segment Anything
    Model (SAM) applied to Yolo detections.

### annotation tools

* [CVAT Automatic Labeling](https://github.com/cvat-ai/cvat?tab=readme-ov-file#deep-learning-serverless-functions-for-automatic-labeling)
* [Label Studio Auto Annotation](https://labelstud.io/guide/labeling#Perform-ML-assisted-labeling-with-interactive-preannotations)
* [Anylabeling](https://github.com/vietanhdev/anylabeling)

## private tools

* [Roboflow Auto Label](https://roboflow.com/auto-label)
* [Encord Data Agent](https://encord.com/data-agents/) 
* [Supervisely NN Image Labeling](https://github.com/supervisely-ecosystem/nn-image-labeling/tree/master)
* [Labellerr LabelGPT](https://www.labellerr.com/labelgpt)
* [Labelbox Annotation With Foundation Model](https://labelbox.com/guides/automatically-label-images-with-99-accuracy-using-foundation-models/)
* [IBM Maximo Visual Insight](https://www.ibm.com/docs/en/visual-insights?topic=tool-automatically-labeling-sample-images)
* [EpigosAi Auto-Label](https://epigos.ai/auto-label)
* [SuperbAi Automated data labeling](https://superb-ai.com/en/products)
* [Amazon SageMaker Automated Labeler](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-automated-labeling.html)
* [Google Image Labeling](https://developers.google.com/ml-kit/vision/image-labeling)
* [Unitlab Auto Data Labeling](https://unitlab.ai/en/data-annotation)
* [Lightly AutoLabeling](https://www.lightly.ai/autolabeling)
* [Landing AI Label Assist](https://support.landing.ai/docs/label-assist)
* [V7 Go](https://www.v7labs.com/go)
* [Sentisight AI assisted image labelling](https://www.sentisight.ai/ai-assisted-image-labeling/)
* [CloudFactory Automated Labeling](https://wiki.cloudfactory.com/docs/userdocs/projects/creating-and-editing-a-project/automated-labeling)
* [Clarifai Automated Data Labeling](https://docs.clarifai.com/guide/auto-labeling)
* [Samsung SDS AutoLabel](https://www.samsungsds.com/us/autolabel/autolabel.html)
* [TrainYolo](https://www.trainyolo.com/)

## scientific publications

### Medical application

* Kim, D., Chung, J., Choi, J., Succi, M. D., Conklin, J., Longo, M. G. F., ... & Do, S. (2022). 
    [Accurate auto-labeling of chest X-ray images based on quantitative similarity to an explainable AI model](https://www.nature.com/articles/s41467-022-29437-8). 
    Nature communications, 13(1), 1867.



