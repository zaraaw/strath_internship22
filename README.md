# Digital Twinning Through Physics Informed Machine Learning. A Vibrating Bearing Case Study

### HIGH LEVEL DESCRIPTION OF SINDy
![SINDy_high_level drawio(1)](https://user-images.githubusercontent.com/63156182/186113939-4cc1aa39-c80b-4b24-a3f7-da9ff1dd14a6.png)

### HOW TO CHOOSE WHAT VERSION OF SINDy
![SINDy drawio](https://user-images.githubusercontent.com/63156182/186113773-0d86bf14-33cc-44a8-b086-c3855e33c762.png)


# Conclusions

#### Comparing SINDy, SINDy-PI, and Ensemble-SINDy applied to the Case Western and NASA datasets

![image](https://user-images.githubusercontent.com/63156182/186119562-07f6ec71-158a-45a7-97d2-842d4cc1c567.png)
![image](https://user-images.githubusercontent.com/63156182/186131974-74ba9b15-7482-4702-81e0-487b547141a4.png)


*See ‘faulty data’ (in notes>logbook files) for more detail on the above results*

##### Conclusions from the above table

SINDy-PI with the NASA Bearing dataset is the best fit model. Abrupt SINDy was not able to be run for either dataset, and therefore cannot be compared to the other SINDy variations. The Case Western Data was underfit for all variations of SINDy. The ensemble-SINDy method subsamples the data randomly, so each time you run the code for this, you may get slightly different equations. 

#### Under-fitting and over-fitting
In pysindy, when defining the regression method, you can define ‘threshold’. This is referred to as the sparsification parameter in the papers. The larger this value, the more sparse the discovered equations are, but if it’s too high the model is underfit. Similarly, if this value is too low the model will be over-fit. 
You also need to consider the candidate libraries, if the library you’ve defined doesn’t have enough terms you’ll get an underfit model. Similarly, if its too large, the model would end up overfit/ the discovered equations would be too busy)


#### SINDy
- SINDy (original) will recover the dynamics in the form of first order differential eqaution 
- It doesnt work well for noisy data (work best with 'synthetic data', as used in the papers)

#### SINDy-PI
- The model has an equation for each term of the candidate library 
  - E.g. if your library is [1, x, x_t, x_tt] then you’ll get 4 equations of the form 1 = …, x = …, x_t = …, and x_tt = …
  - Note this is the only version of SINDy that allows for time-dependant terms in the library (higher order derivative terms, sin(t) etc.)   
- For the NASA bearing dataset the equation “x0 = …” looks the closest to the training data, but “x0_t = …” is the best for the Case Western data 
- Overall SINDy-PI applied to the NASA Bearing Dataset was the best for recovering the dynamcics (out of all combinations of SINDy variation and dataset that were tried)

#### Ensemble-SINDy
- There are 4 types of ensembling (look at example on pysindy gtihub for explanations), V1, V2, V3, and V4
- Remaking the model before fitting it stops it working 
  - E.g. if you’ve fit with V1 ensembling, then remade the model and fit w/ V2 model, it ends up not working  
- V1, V2, and V3 almost always gives the same model, V4 gives a more sparse model (and looks more accurate on NASA bearing dataset) 


#### Further work - tailoring the model
*For context on what the library and optimisation options are, see the literature review*
- This work covers changing the library for each SINDy variation 
  - It was found that SINDy-PI was the only variation that allowed time dependant terms in the library
  - It was also found that for SINDy (original) increasing the number of terms in the library increased the number of terms in the recovered equations
  - Overall, the choice of the library is very important as the wrong choice prevents the method from working at all
 - This work also covered changing the threshold
    - The conclusions from this are described in the section 'Overfitting and Underfitting' above
  
 - Therefore, future work would cover adapting the model further
   - By changing the optimisation method (this work only used Sequentially Thresholded Least Sqaures)
   - By trying different terms in the library (e.g. exponential terms, logarithm terms, etc.)

 - Aswell as adapting the SINDy variations discussed here, there are other SINDy variations (e.g. SINDy-SA, SINDy-C, Abrupt-SINDY (see the literature review)) that could be investigated. 
