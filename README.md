<h1 align="center">Neural View Synthesis and Appearance Editing from Unstructured Images</h1>
<p align="center"><b>Indian Conference on Computer Vision, Graphics and Image Processing</b></p>
<div align="center">
  <span>
    <a href="https://darthgera123.github.io/">Pulkit Gera<sup>1</sup></a>,
    <a href="https://aakashkt.github.io/">Aakash KT<sup>1</sup></a>,
    <a href="https://github.com/thesigmaguy">Dhawal Sirikonda<sup>1</sup></a>,
    <a href="https://scholar.google.co.in/citations?user=3HKjt_IAAAAJ&hl=en">P J Narayanan<sup>1</sup></a>
  </span>
</div>
<p align="center"><sup>1</sup>CVIT, IIIT Hyderabad</p>
<hr>
<img src="main.png" width="900px" height="450px">
<div align="center">
  <span>
    <a href="https://darthgera.github.io/appearance-editing">[Project page]</a>
    <a href="https://dl.acm.org/doi/abs/10.1145/3490035.3490299">[Paper]</a>
    <a href="https://youtu.be/ZCVQj5FK0C4">[Video]</a>
    <a href="https://drive.google.com/file/d/1ky5SRtuPfOxxXgXGkKbhYgpgN-chzWq0/view?usp=sharing">[Data]</a>
    <a href="./bibtex.txt">[bibtex]</a>
  </span>
</div>
<hr>
<p><b>Abstract</b><br>
  We present a neural rendering framework for simultaneous view synthesis and appearance editing of a scene from
  multi-view images captured under known environment illumination. Existing approaches either achieve view synthesis alone or view synthesis along with relighting, without direct control over the scene’s appearance. Our approach explicitly disentangles the appearance and learns a lighting representation that is independent of it. Specifically, we  independently estimate the BRDF and use it to learn a lighting-only representation of the scene. Such disentanglement allows our approach to generalize to arbitrary changes in appearance while performing view synthesis. We show results of editing the appearance of a real scene, demonstrating that our approach produces plausible appearance editing. The performance of our view synthesisapproach is demonstrated to be at par with state-of-the-art
  approaches on both real and synthetic data.
</p>

# Code Instructions
## Prerequisites
This code was tested on UBuntu 20.04, with Python 3.8. For running the code we used `pytorch 3.8`. Please check `requirements.txt` for other dependencies<br>

## Preprocess the Data
Checkout [`preprocess`](./preprocess) for instructions on how to generate and preprocess the data.
## Running code
+ [`DNR`](./DNR) for instructions on how to run DNR code.
+ [`Independent`](./Independent) for instructions on how to run code with independent optimization.
+ [`Joint`](./Joint) for instructions on how to run code with joint optimization.

## Training Data
Contact the authors for Training Data. The 4 datasets : Woman, Plants, Fish and Cushion will be made available soon.
