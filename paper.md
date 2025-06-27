---
title: "Smart-TLC: UX software for semi-automated thin-layer chromatography analysis"
authors:
  - name: Daniella L. Vale
    affiliation: "Department of Analytical Chemistry, Institute of Chemistry, Federal University of Rio de Janeiro, Brazil"
    orcid: 0000-0003-0658-4206
  - name: Rodolfo S. Barboza
    affiliation: "Department of Analytical Chemistry, Institute of Chemistry, Federal University of Rio de Janeiro, Brazil"
corresponding_author:
  name: Daniella L. Vale
  email: daniellavale@iq.ufrj.br
  github: DaniellaVale
date: 2025-06-27
repository: https://github.com/DaniellaVale/SmartTLC_0.1
---

## Summary

Smart-TLC is a Python-based open-source software for semi-automated thin-layer chromatography (TLC) analysis, combining computer vision and machine learning (CNN) techniques. The tool enables image acquisition from a webcam, grayscale-based normalization of chromatographic spots, and quantitative analysis through polynomial regression models to estimate analyte concentrations. Its graphical user interface (developed in Flet) follows user experience (UX) principles, making it accessible even for non-progr...

## Statement of Need

Thin-layer chromatography remains a low-cost, accessible analytical technique but traditionally depends on expensive densitometric instruments. Smart-TLC fills a technological gap by providing:

- a user-friendly interface
- webcam-based image acquisition
- automated normalization of spots
- polynomial quantification
- offline, free, and open-source operation

It empowers researchers and students to perform semi-quantitative and quantitative TLC analysis without requiring advanced programming skills or costly equipment.

## Installation

Requirements:

- Python ≥ 3.10  
- OpenCV  
- TensorFlow  
- Pandas  
- Pillow  
- Flet  
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application on Windows:

```bash
Iniciar_SmartTLC.bat
```

## Features

- Webcam-based TLC plate image capture  
- Coordinate-based cropping and storage of metadata  
- Automatic grayscale normalization  
- Quantitative analysis by quadratic regression  
- Visualization of normalized spot areas in bar charts  
- Modular and intuitive UX-based graphical interface  

## Acknowledgments

This work used generative AI tools (ChatGPT, DeepSeek) to accelerate code structuring, error handling, and manuscript editing, under the full supervision and authorship responsibility of the developers.

## Citation

If you use Smart-TLC in your work, please cite:

```
Vale, D. L., & Barboza, R. S. (2025). Smart-TLC: the UX software application for quantitative TLC analysis. 
```

## License

This project is released under a modified MIT license (for non-commercial use).

## References

- Wall, P. E. (2005). Thin-layer Chromatography: A Modern Practical Approach. Royal Society of Chemistry.  
- Reich, E. and Schibli, A. (2007). High-performance thin-layer chromatography for the analysis of medicinal plants. Journal of AOAC International, 90(2), 408–428.  
- Queiroz, F. et al. (2017). Good Usability Practices in Scientific Software Development. arXiv:1709.00111.  
- Cordero, C. & Vincenti, M. (2024). Advances in Chromatography Using Artificial Intelligence and Machine Learning. ChromatographyOnline. 20(7), 31–34.
