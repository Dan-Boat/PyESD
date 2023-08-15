.. _overview:

.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />

Overview
========
Why is downscaling important?
------------------------------

Downscaling of climate information is crucial because of the widespread and diverse effects of 
human-caused climate change. To better understand climate change impacts, it is essential to generate 
accurate predictions about future climate conditions at a relevant scale for studying its effects and 
creating strategies to address them. General Circulation Models (GCMs) are physics-based numerical models 
that predict future climate patterns and their effects under different assumptions of radiative forcing.
However, they have limitations. While they can replicate many current and past atmospheric processes on
large scales, they struggle with representing smaller-scale processes, like local weather patterns, clouds, 
and certain climate variables, due to their coarse resolution. Additionally, they can't adequately capture local
and regional climate variations. To overcome these limitations, GCM simulations need to be downscaled, 
allowing us to predict regional climates more accurately.

What is the Perfect Prognosis?
------------------------------

Empirical Statistical Downscaling models fall into two categories: Model Output Statistics (MOS)
and Perfect Prognosis (PP). MOS uses GCM data directly to create a model with bias correction 
techniques for downscaling. However, it's inflexible because it's tied to specific GCM products.
On the other hand, PP-ESD trains the downscaling model using weather stations and large-scale
observations like reanalysis products and then connects the trained model to any GCM product 
for predicting downscaled future climates. PP establishes a relationship between larger observed 
patterns and local data, acting as a transfer function for predictions. While PP is more complex
to design and requires substantial modeling, it offers flexibility to work with various data sources.
