
# Awesome Large Scale Aerial Scene Syntheis

In this repo, I'm going to maintain the list of papers related to Large Scale Aerial Scene Syntheis. Specially, I'm going to cover NeRF relevant papers.  


## Papers

Here are some related Papers and Code 

### 2023
* #### CVPR
  - [IGrid-guided Neural Radiance Fields for Large Urban Scenes](https://city-super.github.io/gridnerf/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]]() 
    <details> <summary>Abstract</summary> 
    Target Scenes. In this work, we perform large urban scene rendering with novel grid-guided neural radiance fields. An example of a large urban scene is shown on the left, which spans over 2.7km^2 ground areas captured by over 5k drone images. We show that the rendering results from NeRF-based methods, are blurry and overly smoothed with limited model capacity, while feature grid-based methods tend to display noisy artifacts when adapting to large-scale scenes with high-resolution feature grids. Our proposed two-branch model combines the merits from both approaches and achieves photorealistic novel view renderings with remarkable improvements over existing methods. Both of the two branches gain significant enhancements over their individual baselines. < /details> 
  
  - [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://www.timothybrooks.com/instruct-pix2pix/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://github.com/timothybrooks/instruct-pix2pix) 
    <details> <summary>Abstract</summary> 
     We propose a method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, our model follows these instructions to edit the image. To obtain training data for this problem, we combine the knowledge of two large pretrained models---a language model (GPT-3) and a text-to-image model (Stable Diffusion)---to generate a large dataset of image editing examples. Our conditional diffusion model, InstructPix2Pix, is trained on our generated data, and generalizes to real images and user-written instructions at inference time. Since it performs edits in the forward pass and does not require per-example fine-tuning or inversion, our model edits images quickly, in a matter of seconds. We show compelling editing results for a diverse collection of input images and written instructions. < /details> 


* #### unsorted 
  - [StableLM Suite of Language Models](https://github.com/Stability-AI/StableLM)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models) 
    <details> <summary>Abstract</summary> 
      < /details> 
  
  
  - [Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions](https://instruct-nerf2nerf.github.io/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://github.com/ayaanzhaque/instruct-nerf2nerf) 
    <details> <summary>Abstract</summary> 
     We propose a method for editing NeRF scenes with textinstructions. Given a NeRF of a scene and the collection of images used to reconstruct it, our method uses an imageconditioned diffusion model (InstructPix2Pix) to iteratively edit the input images while optimizing the underlying scene, resulting in an optimized 3D scene that respects the edit instruction. We demonstrate that our proposed method is able to edit large-scale, real-world scenes, and is able to accomplish more realistic, targeted edits than prior work. Result videos can be found on the project website: https://instruct-nerf2nerf.github.io. < /details> 
  
    
  
  
  - [4K-NeRF: High Fidelity Neural Radiance Fields at Ultra High Resolutions](https://arxiv.org/abs/2212.04701)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://github.com/frozoul/4K-NeRF) 
    <details> <summary>Abstract</summary> 
      In this paper, we present a novel and effective framework, named 4K-NeRF, to pursue high fidelity view synthesis on the challenging scenarios of ultra high resolutions, building on the methodology of neural radiance fields (NeRF). The rendering procedure of NeRF-based methods typically relies on a pixel-wise manner in which rays (or pixels) are treated independently on both training and inference phases, limiting its representational ability on describing subtle details, especially when lifting to a extremely high resolution. We address the issue by exploring ray correlation to enhance high-frequency details recovery. Particularly, we use the 3D-aware encoder to model geometric information effectively in a lower resolution space and recover fine details through the 3D-aware decoder, conditioned on ray features and depths estimated by the encoder. Joint training with patch-based sampling further facilitates our method incorporating the supervision from perception oriented regularization beyond pixel-wise loss. Benefiting from the use of geometry-aware local context, our method can significantly boost rendering quality on high-frequency details compared with modern NeRF methods, and achieve the state-of-the-art visual quality on 4K ultra-high-resolution scenario. < /details> 

  - [NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors](https://arxiv.org/abs/2212.03267)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Code]
    <details> <summary>Abstract</summary>  
        2D-to-3D reconstruction is an ill-posed problem, yet humans are good at solving this problem due to their prior knowledge of the 3D world developed over years. Driven by this observation, we propose NeRDi, a single-view NeRF synthesis framework with general image priors from 2D diffusion models. Formulating single-view reconstruction as an image-conditioned 3D generation problem, we optimize the NeRF representations by minimizing a diffusion loss on its arbitrary view renderings with a pretrained image diffusion model under the input-view constraint. We leverage off-the-shelf vision-language models and introduce a two-section language guidance as conditioning inputs to the diffusion model. This is essentially helpful for improving multiview content coherence as it narrows down the general image prior conditioned on the semantic and visual features of the single-view input image. Additionally, we introduce a geometric loss based on estimated depth maps to regularize the underlying 3D geometry of the NeRF. Experimental results on the DTU MVS dataset show that our method can synthesize novel views with higher quality even compared to existing methods trained on this dataset. We also demonstrate our generalizability in zero-shot NeRF synthesis for in-the-wild images. < /details> 

  - [GARF:Geometry-Aware Generalized Neural Radiance Field](https://arxiv.org/abs/2212.02280)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Code]

    <details> <summary>Abstract</summary>  
        Neural Radiance Field (NeRF) has revolutionized free viewpoint rendering tasks and achieved impressive results. However, the efficiency and accuracy problems hinder its wide applications. To address these issues, we propose Geometry-Aware Generalized Neural Radiance Field (GARF) with a geometry-aware dynamic sampling (GADS) strategy to perform real-time novel view rendering and unsupervised depth estimation on unseen scenes without per-scene optimization. Distinct from most existing generalized NeRFs, our framework infers the unseen scenes on both pixel-scale and geometry-scale with only a few input images. More specifically, our method learns common attributes of novel-view synthesis by an encoder-decoder structure and a point-level learnable multi-view feature fusion module which helps avoid occlusion. To preserve scene characteristics in the generalized model, we introduce an unsupervised depth estimation module to derive the coarse geometry, narrow down the ray sampling interval to proximity space of the estimated surface and sample in expectation maximum position, constituting Geometry-Aware Dynamic Sampling strategy (GADS). Moreover, we introduce a Multi-level Semantic Consistency loss (MSC) to assist more informative representation learning. Extensive experiments on indoor and outdoor datasets show that comparing with state-of-the-art generalized NeRF methods, GARF reduces samples by more than 25\%, while improving rendering quality and 3D geometry estimation. < /details> 



### 2022

* #### CVPR

  - [Mega-NERF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs](https://arxiv.org/abs/2112.10703)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://github.com/cmusatyalab/mega-nerf)

    <details> <summary>Abstract</summary>  
        We use neural radiance fields (NeRFs) to build interactive 3D environments from large-scale visual captures spanning buildings or even multiple city blocks collected primarily from drones. In contrast to single object scenes (on which NeRFs are traditionally evaluated), our scale poses multiple challenges including (1) the need to model thousands of images with varying lighting conditions, each of which capture only a small subset of the scene, (2) prohibitively large model capacities that make it infeasible to train on a single GPU, and (3) significant challenges for fast rendering that would enable interactive fly-throughs. To address these challenges, we begin by analyzing visibility statistics for large-scale scenes, motivating a sparse network structure where parameters are specialized to different regions of the scene. We introduce a simple geometric clustering algorithm for data parallelism that partitions training images (or rather pixels) into different NeRF submodules that can be trained in parallel.We evaluate our approach on existing datasets (Quad 6k and UrbanScene3D) as well as against our own drone footage, improving training speed by 3x and PSNR by 12%. We also evaluate recent NeRF fast renderers on top of Mega-NeRF and introduce a novel method that exploits temporal coherence. Our technique achieves a 40x speedup over conventional NeRF rendering while remaining within 0.8 db in PSNR quality, exceeding the fidelity of existing fast renderers. < /details> 

  - [Block-NeRF: Scalable Large Scene Neural View Synthesis](https://waymo.com/research/block-nerf/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Code] 
    
    <details> <summary>Abstract</summary> 
      We present Block-NeRF, a variant of Neural Radiance Fields that can represent large-scale environments. Specifically, we demonstrate that when scaling NeRF to render city-scale scenes spanning multiple blocks, it is vital to decompose the scene into individually trained NeRFs. This decomposition decouples rendering time from scene size, enables rendering to scale to arbitrarily large environments, and allows per-block updates of the environment. We adopt several architectural changes to make NeRF robust to data captured over months under different environmental conditions. We add appearance embeddings, learned pose refinement, and controllable exposure to each individual NeRF, and introduce a procedure for aligning appearance between adjacent NeRFs so that they can be seamlessly combined. We build a grid of Block-NeRFs from 2.8 million images to create the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco. < /details> 

  - [Urban Radiance Fields](https://urban-radiance-fields.github.io/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Code]
   
    <details> <summary>Abstract</summary>  
        The goal of this work is to perform 3D reconstruction and novel view synthesis from data captured by scanning platforms commonly deployed for world mapping in urban outdoor environments (e.g., Street View). Given a sequence of posed RGB images and lidar sweeps acquired by cameras and scanners moving through an outdoor scene, we produce a model from which 3D surfaces can be extracted and novel RGB images can be synthesized. Our approach extends Neural Radiance Fields, which has been demonstrated to synthesize realistic novel images for small scenes in controlled settings, with new methods for leveraging asynchronously captured lidar data, for addressing exposure variation between captured images, and for leveraging predicted image segmentations to supervise densities on rays pointing at the sky. Each of these three extensions provides significant performance improvements in experiments on Street View data. Our system produces state-of-the-art 3D surface reconstructions and synthesizes higher quality novel views in comparison to both traditional methods (e.g.~COLMAP) and recent neural representations (e.g.~Mip-NeRF). < /details> 





* #### ECCV
  - [BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering ](https://city-super.github.io/citynerf/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Code]
    
    <details> <summary>Abstract</summary>  
        Neural Radiance Field (NeRF) has achieved outstanding performance in modeling 3D objects and controlled scenes, usually under a single scale. In this work, we make the first attempt to bring NeRF to city-scale, with views ranging from satellite-level that captures the overview of a city, to ground-level imagery showing complex details of an architecture. The wide span of camera distance to the scene yields multi-scale data with different levels of detail and spatial coverage, which casts great challenges to vanilla NeRF and biases it towards compromised results. To address these issues, we introduce CityNeRF, a progressive learning paradigm that grows the NeRF model and training set synchronously. Starting from fitting distant views with a shallow base block, as training progresses, new blocks are appended to accommodate the emerging details in the increasingly closer views. The strategy effectively activates high-frequency channels in the positional encoding and unfolds more complex details as the training proceeds. We demonstrate the superiority of CityNeRF in modeling diverse city-scale scenes with drastically varying views, and its support for rendering views in different levels of detail. < /details> 

* #### TPAMI
    - [Neural Radiance Fields from Sparse RGB-D Images for High-Quality View Synthesis](https://ieeexplore.ieee.org/abstract/document/9999509)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  [Code]
        
         <details> <summary>Abstract</summary> 
            The recently proposed neural radiance fields (NeRF) use a continuous function formulated as a multi-layer perceptron (MLP) to model the appearance and geometry of a 3D scene. This enables realistic synthesis of novel views, even for scenes with view dependent appearance. Many follow-up works have since extended NeRFs in different ways. However, a fundamental restriction of the method remains that it requires a large number of images captured from densely placed viewpoints for high-quality synthesis and the quality of the results quickly degrades when the number of captured views is insufficient. To address this problem, we propose a novel NeRF-based framework capable of high-quality view synthesis using only a sparse set of RGB-D images, which can be easily captured using cameras and LiDAR sensors on current consumer devices. First, a geometric proxy of the scene is reconstructed from the captured RGB-D images. Renderings of the reconstructed scene along with precise camera parameters can then be used to pre-train a network. Finally, the network is fine-tuned with a small number of real captured images. We further introduce a patch discriminator to supervise the network under novel views during fine-tuning, as well as a 3D color prior to improve synthesis quality. We demonstrate that our method can generate arbitrary novel views of a 3D scene from as few as 6 RGB-D images. Extensive experiments show the improvements of our method compared with the existing NeRF-based methods, including approaches that also aim to reduce the number of input images. < /details> 



### 2021


* #### CVPR





* #### ICCV
  * [KiloNeRFSpeeding up Neural Radiance Fields with Thousands of Tiny MLPs](https://creiser.github.io/kilonerf/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://github.com/creiser/kilonerf)

    <details> <summary>Abstract</summary>  
      NeRF synthesizes novel views of a scene with unprecedented quality by fitting a neural radiance field to RGB images. However, NeRF requires querying a deep Multi-Layer Perceptron (MLP) millions of times, leading to slow rendering times, even on modern GPUs. In this paper, we demonstrate that real-time rendering is possible by utilizing thousands of tiny MLPs instead of one single large MLP. In our setting, each individual MLP only needs to represent parts of the scene, thus smaller and faster-to-evaluate MLPs can be used. By combining this divide-and-conquer strategy with further optimizations, rendering is accelerated by three orders of magnitude compared to the original NeRF model without incurring high storage costs. Further, using teacher-student distillation for training, we show that this speed-up can be achieved without sacrificing visual quality. < /details> 

   * [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields ](https://paperswithcode.com/paper/mip-nerf-a-multiscale-representation-for-anti)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://github.com/bebeal/mipnerf-pytorch)

        <details> <summary>Abstract</summary>  The rendering procedure used by neural radiance fields (NeRF) samples a scene with a single ray per pixel and may therefore produce renderings that are excessively blurred or aliased when training or testing images observe scene content at different resolutions. The straightforward solution of supersampling by rendering with multiple rays per pixel is impractical for NeRF, because rendering each ray requires querying a multilayer perceptron hundreds of times. Our solution, which we call "mip-NeRF" (a la "mipmap"), extends NeRF to represent the scene at a continuously-valued scale. By efficiently rendering anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable aliasing artifacts and significantly improves NeRF's ability to represent fine details, while also being 7% faster than NeRF and half the size. Compared to NeRF, mip-NeRF reduces average error rates by 17% on the dataset presented with NeRF and by 60% on a challenging multiscale variant of that dataset that we present. Mip-NeRF is also able to match the accuracy of a brute-force supersampled NeRF on our multiscale dataset while being 22x faster. < /details> 
 





* #### ECCV




* #### Others
    * [DeRF: Decomposed Radiance Fields](https://ubc-vision.github.io/derf/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  [[Code]](https://github.com/ubc-vision/derf/)
        <details> <summary>Abstract</summary>  
            With the advent of Neural Radiance Fields (NeRF), neural networks can now render novel views of a 3D scene with quality that fools the human eye. Yet, generating these images is very computationally intensive, limiting their applicability in practical scenarios. In this paper, we propose a technique based on spatial decomposition capable of mitigating this issue. Our key observation is that there are diminishing returns in employing larger (deeper and/or wider) networks. Hence, we propose to spatially decompose a scene and dedicate smaller networks for each decomposed part. When working together, these networks can render the whole scene. This allows us near-constant inference time regardless of the number of decomposed parts. Moreover, we show that a Voronoi spatial decomposition is preferable for this purpose, as it is provably compatible with the Painter’s Algorithm for efficient and GPU-friendly rendering. Our experiments show that for real-world scenes, our method provides up to 3x more efficient inference than NeRF (with the same rendering quality), or an improvement of up to 1.0~dB in PSNR (for the same inference cost). < /details> 
 
  



### Other



  
