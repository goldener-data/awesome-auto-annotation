# awesome-auto-annotation

When training a new AI pipeline, the annotation process can be painful and quite costly. In most of the projects, 
it is one of the main bottleneck in the release of impactful AI pipelines. Then, leveraging existing AI pipelines to 
automatically generate new annotations is a way to accelerate this process.

This repository is aiming to provide a curated list of existing resources demonstrating automatic data labeling. 
The information is split by data type (image, text, video, sound, ...) and each of this type is split between:
* list of public repositories
* list of private tools
* list of scientific publications

Warning, our intent here is to provide a list of sources claiming to be able to automatically generate annotations.
Not all of these sources have been tested/verified and there is no guarantee that the claim of the authors are valid.

# image-auto-annotation

## public repositories

### backend only

* [autodistill/autodistill](https://github.com/autodistill/autodistill): Locally annotate from a foundation model and train
    a specific model on top of it.
* [ultralytics/ultralytics](https://docs.ultralytics.com/models/sam-2/#sam-2-comparison-vs-yolo): Auto annotate from Segment Anything
    Model (SAM) applied to Yolo detections.
* [Nvidia Tao Auto Label](https://docs.nvidia.com/tao/tao-toolkit/text/data_services/auto-label.html): run GroundingDINO 
    to annotate images (local or cloud).
* [yasho191/SwiftAnnotate](https://github.com/yasho191/SwiftAnnotate): Generate multimodal annotations using main LLMs and VLMs (private and public ones).
* [AlvaroCavalcante/auto_annotate](https://github.com/AlvaroCavalcante/auto_annotate): Locally annotate from a pretrained 
    object detection model after setting up a threshold.
* [mailcorahul/auto_labeler](https://github.com/mailcorahul/auto_labeler): CLI tools allowing to leverage different
    foundation models to generate annotations on local data.
* [akuonj/omanAI](https://github.com/akuonj/omanAI): Running Yolov8 to add bounding boxes to images.
* [fexploit/ComfyUI-AutoLabel](https://github.com/fexploit/ComfyUI-AutoLabel): generate text description from images.
* [Antraxmin/AutoLabeler](https://github.com/Antraxmin/AutoLabeler): Use Ultralytics to generate annotations on images.
* [pollyjuice74/CVAT-Auto-Labeler](https://github.com/pollyjuice74/CVAT-Auto-Labeler): Use Ultralytics to generate annotations
    on images coming from CVAT.
* [Daniele-Cannella/auto_labeling](https://github.com/Daniele-Cannella/auto_labeling): Use a Phi 3 and grounding to make new annotations on images.
* [jamesnatulan/auto-image-labeler](https://github.com/jamesnatulan/auto-image-labeler): Use foundation models to make zero shot annotations in images.
* [Ziad-Algrafi/ODLabel](https://github.com/Ziad-Algrafi/ODLabel/tree/main?tab=readme-ov-file): Label new images by applying Yolo-World model.
* [sah4jpatel/AutoLabeller](https://github.com/sah4jpatel/AutoLabeller): Label new images by applying Yolo-World model.
* [aiXander/CLIP_assisted_data_labeling](https://github.com/aiXander/CLIP_assisted_data_labeling): Few shot learning to add label from clip embeddings.
* [ChunmingHe/WS-SAM](https://github.com/ChunmingHe/WS-SAM): Generate full segmentation mask in medical images from weak annotations and SAM. 
* [StriveZs/ALPS](https://github.com/StriveZs/ALPS): Leverage SAM to generate annotations on remote sensing images.
* [Fudan-ProjectTitan/OpenAnnotate3D](https://github.com/Fudan-ProjectTitan/OpenAnnotate3D): Open-vocabulary auto-labeling system for multi-modal 3D data.

### annotation tools

* [CVAT Automatic Labeling](https://github.com/cvat-ai/cvat?tab=readme-ov-file#deep-learning-serverless-functions-for-automatic-labeling): General computer vision annotator
* [Label Studio Auto Annotation](https://labelstud.io/guide/labeling#Perform-ML-assisted-labeling-with-interactive-preannotations): General computer vision annotator
* [Anylabeling](https://github.com/vietanhdev/anylabeling): General computer vision annotator
* [fabrylab/clickpoints](https://github.com/fabrylab/clickpoints): General computer vision annotator
* [halleewong/ScribblePrompt](https://github.com/halleewong/ScribblePrompt): User interface to auto-annotate medical images from click prompting.
* [filonenkoa/autoannotator](https://github.com/filonenkoa/autoannotator): User interface with human face automatic annotations
* [Babyhamsta/Yoable](https://github.com/Babyhamsta/Yoable): User interface with YoloV8 automatic annotations
* [FennelFetish/qapyq](https://github.com/FennelFetish/qapyq): User interface allowing to add captions to images using vision language models, and generate bounding boxes from Yolo.
* [anujgaut/All-in-One-Label](https://github.com/anujgaut/All-in-One-Label/tree/main): User interface allowing to run YOLO or Mask R-CNN to generate annotations on images.
* [Ashish-Yallapragada/AI_AssistedImageSegmentation](https://github.com/Ashish-Yallapragada/AI_AssistedImageSegmentation): User interface designed to help the annotation of zebrafish images.
* [ParthShethSK/LabelPal](https://github.com/ParthShethSK/LabelPal): user interface allowing to generate annotations from specific tensorflow models (coco-ssd).
* [ksugar/qupath-extension-sam](https://github.com/ksugar/qupath-extension-sam?tab=readme-ov-file): Use SAM to label medical images in QuPath.

## private tools

* [Roboflow Auto Label](https://roboflow.com/auto-label)
* [Encord Data Agent](https://encord.com/data-agents/) 
* [Supervisely NN Image Labeling](https://github.com/supervisely-ecosystem/nn-image-labeling/tree/master)
* [Picselia Model Assisted Labeling ](https://www.picsellia.com/labeling-tool)
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
* [Intel Geti](https://docs.geti.intel.com/docs/user-guide/geti-fundamentals/annotations/annotation-tools#interactive-segmentation-tool)
* [Digit7 DigitSquare](https://www.digit7.ai/digitsquare/)
* [Kili Technology Model Assisted Labeling](https://kili-technology.com/platform/label-annotate/image-annotation-tool?utm_term=image%20labeling%20tool%20for%20object%20detection&utm_campaign=Kili+-+SN+-+Europe+-+Annotation+Tool&utm_source=adwords&utm_medium=ppc&hsa_acc=4040516345&hsa_cam=20545681372&hsa_grp=179135420138&hsa_ad=746838268298&hsa_src=g&hsa_tgt=kwd-834599580468&hsa_kw=image%20labeling%20tool%20for%20object%20detection&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=20545681372&gbraid=0AAAAACg5iBN5MYcadsCB8aak2tPk12UP7&gclid=Cj0KCQjww-HABhCGARIsALLO6XxbD_FwKelAOjd1lhzc1nDZVGXEj_znnCWA4-cNJiw97CwgOzh55pMaAgLvEALw_wcB)
* [Labelme Automated Annotation](https://labelme.io/)
* [Edge Impulse AI Labeling](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition/ai-labeling)
* [Segments AI Model-assisted labelling](https://docs.segments.ai/tutorials/model-assisted-labeling)
* [AI Wakforce AI assisted labeling](https://www.aiwakforce.com/ai-assisted-labeling-services/)
* [LabelAI AI assisted labeling](https://www.labelai.co/image-annotation)
* [Deepen AI Auto-Label](https://www.deepen.ai/image-annotation)
* [DataGym AI-assisted pre-labeling](https://docs.datagym.ai/documentation/ai-assistant/ai-assisted-pre-labeling)
* [JTheta Auto-Label](https://www.jtheta.ai/mlmodelassistedlabeling)
* [Halcon AI-Assisted Labeling](https://halcon.ai/vision-annotation.html)
* [BasicAI Auto Annotation](https://www.basic.ai/basicai-cloud-data-annotation-platform/ai-data-annotation-toolset)

## scientific publications

### Survey

* Mots'oehli, Moseli. 
    [Assistive Image Annotation Systems with Deep Learning and Natural Language Capabilities: A Review.](https://arxiv.org/html/2407.00252v1)
    2024 International Conference on Emerging Trends in Networks and Computer Communications (ETNCC). IEEE, 2024.

### General application

* Toubal, Imad Eddine, et al. 
    [Modeling collaborator: Enabling subjective vision classification with minimal human effort via llm tool-use.](https://openaccess.thecvf.com/content/CVPR2024/html/Toubal_Modeling_Collaborator_Enabling_Subjective_Vision_Classification_With_Minimal_Human_Effort_CVPR_2024_paper.html)
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Multimodal application

* Zhou, Yijie, et al. 
    [Openannotate3d: Open-vocabulary auto-labeling system for multi-modal 3d data.](https://ieeexplore.ieee.org/abstract/document/10610779)
    2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024.
* Gallagher, James E.,et al. 
    [A Multispectral Automated Transfer Technique (MATT) for machine-driven image labeling utilizing the Segment Anything Model (SAM).] (https://ieeexplore.ieee.org/abstract/document/10815733)
    IEEE Access (2024).

### Medical application

* Kim, Doyun, et al. 
    [Accurate auto-labeling of chest X-ray images based on quantitative similarity to an explainable AI model.](https://www.nature.com/articles/s41467-022-29437-8). 
    Nature communications 13.1 (2022): 1867.
* Diaz-Pinto, Andres, et al. 
    [Monai label: A framework for ai-assisted interactive labeling of 3d medical images.](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001324)
    Medical Image Analysis 95 (2024): 103207.
* Deshpande, Tanvi, et al. 
    [Auto-Generating Weak Labels for Real & Synthetic Data to Improve Label-Scarce Medical Image Segmentation.](https://arxiv.org/abs/2404.17033) 
    arXiv preprint arXiv:2404.17033 (2024).
* He, Chunming, et al. 
    [Weakly-supervised concealed object segmentation with sam-based pseudo labeling and multi-scale feature grouping.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/61aa557643ae8709b6a4f41140b2234a-Abstract-Conference.html)
    Advances in Neural Information Processing Systems 36 (2023): 30726-30737.
* Cazzaniga, Giorgio, et al. 
    [Improving the annotation process in computational pathology: A pilot study with manual and semi-automated approaches on consumer and medical grade devices.](https://link.springer.com/article/10.1007/s10278-024-01248-x)
    Journal of Imaging Informatics in Medicine 38.2 (2025): 1112-1119.


### Environmental monitoring application
* Zurowietz, Martin, et al. 
    [MAIA—A machine learning assisted image annotation method for environmental monitoring and exploration.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207498)
    PloS one 13.11 (2018): e0207498.

### Remote sensing application

* Zhang, Song, et al. 
    [Alps: An auto-labeling and pre-training scheme for remote sensing segmentation with segment anything model.](https://ieeexplore.ieee.org/abstract/document/10949707) 
    IEEE Transactions on Image Processing (2025).

## Deprecated repositories

In this section, we are listing code repositories with less than 1 year activity.

* [virajmavani/semi-auto-image-annotation-tool](https://github.com/virajmavani/semi-auto-image-annotation-tool)
* [mdhmz1/Auto-Annotate](https://github.com/mdhmz1/Auto-Annotate)
* [demelere/Geospatial-auto-segmentation-and-labelling](https://github.com/demelere/Geospatial-auto-segmentation-and-labelling)