# Face Generation Methods

## Contents
- [Basic Networks](#Basic-Networks)
- [Face Swap](#Face-Swap)
- [Face Reenactment](#Faace-Reenactment)
- [Datasets](#Datasets)

---
### Basic Networks
#### 2022

#### 2021
- [CVPR'21] [[**pSp**](https://openaccess.thecvf.com/content/CVPR2021/papers/Richardson_Encoding_in_Style_A_StyleGAN_Encoder_for_Image-to-Image_Translation_CVPR_2021_paper.pdf)] Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation [[pytorch](https://github.com/eladrich/pixel2style2pixel)]
- [arXiv'21.06] [[**AttnFlow**](https://arxiv.org/pdf/2106.03959.pdf)] Generative Flows with Invertible Attentions
- [arXiv'21.04] [[**StyleGAN-Inversion**](https://arxiv.org/pdf/2104.07661.pdf)] A Simple Baseline for StyleGAN Inversion [[web](https://wty-ustc.github.io/inversion/)] [[pytorch](https://github.com/bes-dev/MobileStyleGAN.pytorch)]
#### 2020 
- [ICML'20] [[**AGD**](https://arxiv.org/pdf/2006.08198.pdf)] AutoGAN-Distiller: Searching to Compress Generative Adversarial Networks [[pytorch](https://github.com/VITA-Group/AGD)]
- [NIPS'20] [[**ContraGAN**](https://proceedings.neurips.cc//paper/2020/file/f490c742cd8318b8ee6dca10af2a163f-Paper.pdf)] ContraGAN: Contrastive Learning for Conditional Image Generation  [[tensorflow](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)]
- [NIPS'20] [[**CircleGAN**](https://papers.nips.cc/paper/2020/file/f14bc21be7eaeed046fed206a492e652-Paper.pdf)] CircleGAN: Generative Adversarial Learning across Spherical Circles  [[tensorflow](https://github.com/POSTECH-CVLab/circlegan)]
- [NIPS'20] [[**DeepI2I**](https://proceedings.neurips.cc/paper/2020/file/88855547570f7ff053fff7c54e5148cc-Paper.pdf)] DeepI2I: Enabling Deep Hierarchical Image-to-Image Translation by Transferring from GANs  [[pytorch](https://github.com/yaxingwang/DeepI2I)]
- [NIPS'20] [[**NVAE**](https://proceedings.neurips.cc/paper/2020/file/e3b21256183cf7c2c7a66be163579d37-Paper.pdf)] NVAE: A Deep Hierarchical Variational Autoencoder  [[pytorch](https://github.com/NVlabs/NVAE)]
- [NIPS'20] [[**Swapping-Autoencoder**](https://proceedings.neurips.cc/paper/2020/file/50905d7b2216bfeccb5b41016357176b-Paper.pdf)] Swapping Autoencoder for Deep Image Manipulation  [[web](https://taesung.me/SwappingAutoencoder/)] [[pytorch](https://github.com/taesungp/swapping-autoencoder-pytorch)]
- [ECCV'20] [[**COCO-FUNIT**](https://nvlabs.github.io/COCO-FUNIT/paper.pdf)] COCO-FUNIT: Few-Shot Unsupervised Image Translation with a Content Conditioned Style Encoder  [[web](https://nvlabs.github.io/COCO-FUNIT/)] [[pytorch](https://github.com/NVlabs/imaginaire/blob/master/projects/coco_funit/README.md)]
- [ECCV'20] [[**TopologyGAN**](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480120.pdf)] TopologyGAN: Topology Optimization Using Generative Adversarial Networks Based on Physical Fields Over the Initial Domain  [[pytorch](https://github.com/basiralab/topoGAN)]
- [ECCV'20] [[**wc-Vid2Vid**](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530358.pdf)] World-Consistent Video-to-Video Synthesis  [[web](https://nvlabs.github.io/wc-vid2vid/)] [[pytorch](https://github.com/NVlabs/imaginaire/blob/master/projects/wc_vid2vid/README.md)]
- [CVPR'20] [[**StarGAN2**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.pdf)] StarGAN v2: Diverse Image Synthesis for Multiple Domains  [[pytorch](https://github.com/clovaai/stargan-v2)]

#### 2019
- [NeurIPS'19] [[**fs-Vid2Vid**](https://openreview.net/pdf?id=rkluKVrl8H)] Few-shot Video-to-Video Synthesis  [[web](https://nvlabs.github.io/few-shot-vid2vid/)] [[pytorch](https://github.com/NVlabs/few-shot-vid2vid)]
- [ICCV'19] [[**FUNIT**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Few-Shot_Unsupervised_Image-to-Image_Translation_ICCV_2019_paper.pdf)] Few-Shot Unsupervised Image-to-Image Translation  [[web](https://nvlabs.github.io/FUNIT/)] [[pytorch](https://github.com/NVlabs/FUNIT)]
- [CVPR'19] [[**SPADE**](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.pdf)] Semantic Image Synthesis with Spatially-Adaptive Normalization  [[web](https://nvlabs.github.io/SPADE/)] [[pytorch](https://github.com/nvlabs/spade/)]
#### 2018
- [NeurIPS'18] [[**Glow**](https://papers.nips.cc/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf)] Glow: Generative Flow with Invertible 1x1 Convolutions  [[pytorch](https://github.com/chaiyujin/glow-pytorch)] [[tensorflow](https://github.com/openai/glow)]
- [NeurIPS'18] [[**Vid2Vid**](https://papers.nips.cc/paper/2018/file/d86ea612dec96096c5e0fcc8dd42ab6d-Paper.pdf)] Video-to-Video Synthesis  [[web](https://tcwang0509.github.io/vid2vid/)] [[pytorch](https://github.com/NVIDIA/vid2vid)]
- [CVPR'18] [[**StarGAN**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf)] StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation  [[pytorch](https://github.com/yunjey/stargan)]
- [CVPR'18] [[**Pix2PixHD**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.pdf)] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs  [[web](https://tcwang0509.github.io/pix2pixHD/)] [[pytorch](https://github.com/NVIDIA/pix2pixHD)]
- [ECCV'18] [[**MUNIT**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf)] Multimodal Unsupervised Image-to-Image Translation  [[pytorch](https://github.com/NVlabs/MUNIT)]
- [NeurIPS'17] [[**UNIT**](https://papers.nips.cc/paper/2017/file/dc6a6489640ca02b0d42dabeb8e46bb7-Paper.pdf)] Unsupervised Image-to-Image Translation Networks  [[pytorch](https://github.com/mingyuliutw/UNIT)]
#### 2017
- [ICCV'17] [[**CycleGAN**](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  [[web](https://junyanz.github.io/CycleGAN/)] [[pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]
- [CVPR'17] [[**Pix2Pix**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)] Image-to-Image Translation with Conditional Adversarial Networks  [[web](https://phillipi.github.io/pix2pix/)] [[pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]
- 
---
### Face Swap
#### 2022 
- [PAMI'22][[**FSGANv2**](https://arxiv.org/pdf/2202.12972.pdf)] FSGANv2: Improved Subject Agnostic Face Swapping and Reenactment 

#### 2021
- [IJCAI'21] [[**HifiFace**](https://arxiv.org/pdf/2106.09965.pdf)] HifiFace: 3D Shape and Semantic Prior Guided High Fidelity Face Swapping  [[home](https://johann.wang/HifiFace/)] [[unofficial pytorch](https://github.com/mindslab-ai/hififace)]
- [CVPR'21] [[**HFaceInpainter**](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_FaceInpainter_High_Fidelity_Face_Adaptation_to_Heterogeneous_Domains_CVPR_2021_paper.pdf)] HFaceInpainter: High Fidelity Face Adaptation to Heterogeneous Domains 
- [CVPR'21] [[**MegaFS**](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_One_Shot_Face_Swapping_on_Megapixels_CVPR_2021_paper.pdf)] One Shot Face Swapping on Megapixels [] [[pytorch](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)]

#### 2020
- [CVPR'20] [[**Faceshifter**](https://arxiv.org/pdf/1912.13457.pdf)] FaceShifter: Towards High Fidelity And Occlusion Aware Face Swapping  [[home](https://lingzhili.com/FaceShifterPage/)][[unoffical pytorch](https://github.com/mindslab-ai/faceshifter)]
- [SIGGRAPH'20] [[**IDDis**](https://arxiv.org/pdf/2005.07728.pdf)] Face Identity Disentanglement via Latent Space Mapping  [[pytorch](https://github.com/YotamNitzan/ID-disentanglement)]
- [AAAI'20] [[**Facecontroller**](https://arxiv.org/pdf/2102.11464.pdf)] FaceController: Controllable Attribute Editing for Face in the Wild 
- [ACCV'20] [[**UnifiedSR**](https://openaccess.thecvf.com/content/ACCV2020/papers/Le_Minh_Ngo_Unified_Application_of_Style_Transfer_for_Face_Swapping_and_Reenactment_ACCV_2020_paper.pdf)] Unified Application of Style Transfer for Face Swapping and Reenactment 
- [ACM'20] [[**SimSwap**](https://dl.acm.org/doi/pdf/10.1145/3394171.3413630)] SimSwap: An Efficient Framework For High Fidelity Face Swapping  [[pytorch](https://github.com/neuralchen/SimSwap)]
- [arXiv'20] [[**DeepFaceLab**](https://arxiv.org/pdf/2005.05535.pdf)] DeepFaceLab: Integrated, flexible and extensible face-swapping framework 

#### 2019
- [ICCV'19] [[**FSGAN**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nirkin_FSGAN_Subject_Agnostic_Face_Swapping_and_Reenactment_ICCV_2019_paper.pdf)] FSGAN: Subject Agnostic Face Swapping and Reenactment  [[pytorch](https://github.com/YuvalNirkin/fsgan)]

#### 2018
- [arXiv'18] [[**RSGAN**](https://arxiv.org/pdf/1804.03447.pdf)] RSGAN: Face Swapping and Editing using Face and Hair Representation in Latent Spaces 
- [CVPR'18] [[**OIPFS**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bao_Towards_Open-Set_Identity_CVPR_2018_paper.pdf)] Towards Open-Set Identity Preserving Face Synthesis 
- [CVPR'18] [[**IPNet**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bao_Towards_Open-Set_Identity_CVPR_2018_paper.pdf)] Towards Open-Set Identity Preserving Face Synthesis 
- [ACCV'18] [[**FSNet**](https://arxiv.org/pdf/1811.12666.pdf)] FSNet: An Identity-Aware Generative Model for Image-based Face Swapping  [[web](https://tatsy.github.io/projects/fsnet/)]


---
### Face Reenactment
#### 2022
- [arXiv'22] [[**StyleHEAT**](https://arxiv.org/pdf/2203.04036.pdf)] StyleHEAT: One-Shot High-Resolution Editable Talking Face
Generation via Pre-trained StyleGAN [[home](https://feiiyin.github.io/StyleHEAT/)] [[code](https://github.com/FeiiYin/StyleHEAT/)]
- [ICLR'22] [[**LIA**](https://openreview.net/pdf?id=7r6kDq0mK_)]Latent Image Animator: Learning to Animate Images via Latent Space Navigation [[home](https://wyhsirius.github.io/LIA-project/)] 


#### 2021
- [3DV'21] [[**SAFA**](https://arxiv.org/pdf/2111.04928.pdf)] SAFA: Structure Aware Face Animation [[pytorch](https://arxiv.org/pdf/2111.04928.pdf)]
- [CVPR'21] [[**face-vid2vid**](https://nvlabs.github.io/face-vid2vid/main.pdf)] One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing  [[web](https://nvlabs.github.io/face-vid2vid/)]
- [ICCV'21] [[**HeadGAN**](https://arxiv.org/pdf/2012.08261.pdf)] HeadGAN: One-shot Neural Head Synthesis and Editing [[home](https://michaildoukas.github.io/HeadGAN/)] [[video](https://www.youtube.com/watch?v=Xo9IW3cMGTg)]
- [ICCV'21] [[**PIRenderer**](https://openaccess.thecvf.com/content/ICCV2021/papers/Ren_PIRenderer_Controllable_Portrait_Image_Generation_via_Semantic_Neural_Rendering_ICCV_2021_paper.pdf)] PIRenderer: Controllable Portrait Image Generation via Semantic Neural
Rendering [[home](https://renyurui.github.io/PIRender_web/)] [[video](https://www.youtube.com/watch?v=gDhcRcPI1JU)] [[pytorch](https://github.com/RenYurui/PIRender)]

#### 2020
- [CVPR'20] [[**FReeNet**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_FReeNet_Multi-Identity_Face_Reenactment_CVPR_2020_paper.pdf)] FReeNet: Multi-Identity Face Reenactment  [[pytorch](https://github.com/zhangzjn/FReeNet)]
- [ICASSP'20] [[**APB2Face**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9052977&tag=1)] APB2Face: Audio-guided face reenactment with auxiliary pose and blink signals  [[pytorch](https://github.com/zhangzjn/APB2Face)]
- [ECCV'20] [[**Bi-layer**](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570511.pdf)] Fast Bi-layer Neural Synthesis of One-Shot Realistic Head Avatars  [[pytorch](https://github.com/saic-violet/bilayer-model)]
- [arXiv'20] [[**FaR-GAN**](https://arxiv.org/pdf/2005.06402.pdf)] FaR-GAN for One-Shot Face Reenactment 
- [AAAI'20] [[**DAE-GAN**](https://ojs.aaai.org/index.php/AAAI/article/view/6970/6824)] Realistic Face Reenactment via Self-Supervised Disentangling of Identity and Pose 


#### 2019 
- [NIPS'19] [[**FOMM**](https://proceedings.neurips.cc/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf)] First Order Motion Model for Image Animation  [[web](https://aliaksandrsiarohin.github.io/first-order-model-website/)] [[pytorch](https://github.com/AliaksandrSiarohin/first-order-model)]
- [BMVC'19] [[**OSFR**](https://arxiv.org/pdf/1908.03251.pdf)] One-shot Face Reenactment  [[home](https://wywu.github.io/projects/ReenactGAN/OneShotReenact.html)] [[pytorch](https://github.com/bj80heyue/One_Shot_Face_Reenactment)]


#### 2018
- [ECCV'18] [[**X2Face**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Olivia_Wiles_X2Face_A_network_ECCV_2018_paper.pdf)] X2Face: A network for controlling face generation by using images, audio, and pose codes  [[pytorch](https://github.com/oawiles/X2Face)]
- [ECCV'18] [[**ReenactGAN**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wayne_Wu_Learning_to_Reenact_ECCV_2018_paper.pdf)] ReenactGAN: Learning to Reenact Faces via Boundary Transfer  [[pytorch](https://github.com/wywu/ReenactGAN)]


---
### Datasets
- [[**CelebA**](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)]
- [[**CelebA-HQ**](https://github.com/tkarras/progressive_growing_of_gans)]
- [[**CelebAMask-HQ**](https://github.com/switchablenorms/CelebAMask-HQ)]
- [[**CelebA-Spoof**](https://github.com/Davidzhangyuanhan/CelebA-Spoof)]
- [[**FFHQ**](https://github.com/NVlabs/ffhq-dataset)]
- [[**VoxCeleb1**](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)]
- [[**VoxCeleb2**](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)]
- [[**VGGFace**](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/)]
- [[**VGGFace2**](https://github.com/ox-vgg/vgg_face2)]
- [[**RaFD**](http://www.socsci.ru.nl:8180/RaFD2/RaFD)]
- [[**Multi-PIE**](https://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html)]
- [[**FaceForensics++**](https://github.com/ondyari/FaceForensics)][[**1000 HifiFace FaceForensics++ **](https://johann.wang/HifiFace/)]
- [[**SCUT-FBP**](https://link.zhihu.com/?target=https%3A//github.com/HCIILAB/SCUT-FBP5500-Database-Release)]
- [[**MakeUp**](http://www.antitza.com/makeup-datasets.html)]


---
Borrow and Thanks for these repos [[awesome-face-generation](https://github.com/zhangzjn/awesome-face-generation)],[[Human Video Generation](https://github.com/yule-li/Human-Video-Generation)]
