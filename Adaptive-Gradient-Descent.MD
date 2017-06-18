# Least Squares Parameter Estimation: Adpative Gradient Descent

This blog post investigates adaptive gradient descent methods and compares it to the closed form solution. This dataset comes from Coursera's Machine Learning course. It aims to put together different methods and also to improve the speed on the algorithm.

#### There are different ways to estimate parameters

* Sickit Learn Package
* Gradient Descent Algorithm

There are 17383 observations in the train data set! I suspect this will slow down the gradient descent quite a bit.

#### Sckitlearn

This software package can easily minimize least square error between data points by solution the the closed form solution of the gradient's derivative.

#### Fitted Model
* predictors~ sqft_living, bedrooms, bathrooms, bed_bath_rooms, bedrooms_squared
* target~ price


        SSE:  1.14723203834e+15

        intercept:  321022.818566

        coefficients:  [    308.16971104 -127863.12577131 -110920.90983384   34641.50131801
         -1390.73582348]

        #### Coefficients
        1. [ 321022.818566]
        2. [ 308.16971104]
        3. [-127863.12577131]
        4. [-110920.90983384]
        5. [34641.50131801]
        6. [-1390.73582348]

The results of actual vs predicted value can be seen in the plot above.

#### Custom Gradient Descent Algorithm

1. Adadelta (a customization of Adagrad) [1]
  * It adapts the learning rate to the parameters
  * Reduces its decreasing learning rate
2. Recursively runs gradient descent
  * Back propogate the parameters
  * Increases learning rate if cost is decreasing
  * Decrease learning rate if cost is increasing and cost is better than average
  * Stop the descent if the decrease in cost is linear



#### What if the learning rate is still too slow?
* Sometimes the gradient doesn't converge to zero easily.
* Getting pretty close can be better than waiting for the algorithm to converge
* It might not be converging to a closed form solution

        #### Coefficients:
        1.	[  3.10436436e+05]
        2.	[  3.07973283e+02]
        3.	[ -1.22504131e+05]
        4.	[ -1.09997997e+05]
        5.	[  3.43850026e+04]
        6.	[ -1.98562365e+03]

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22962104/7becde04-f310-11e6-9e7f-c7a41d318478.png" >
</p>

#### The parameters are close to the closed-form solution
Since the parameters do not converged, the algorithm stops at the maximum iteration which is 9999. In addition, the algorithm takes a whopping 170 seconds


<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22962109/818524fc-f310-11e6-9132-fe5d6c06d95f.png" >
</p>

#### Looking at last 999 epochs
The cost function seems to be decreasing almost linearly while the alpha and gradient functions are decreasing more jaggedly.


#### Save time at the expense of a little accuracy?
The cost function can be check for linearity. The idea is that when cost start decreasing linearly, the algorithm will take too long to converge. It could be beneficial to stop the algorithm when there is limited improvement to the cost function.

        #### Coefficients:
        1.	 [ 298134.76310869]
        2.	 [    307.74037306]
        3.	 [-116591.24620479]
        4.	 [-108423.57186844]
        5.	 [  33957.73138586]
        6.	 [  -2599.27733704]

#### Not too shabby...
The process took 133.72099685668945 seconds and around 7600 iterations. Overall, the parameters are still pretty close and about 40 seconds were shaved off the execution time.

<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22962117/885e8f0c-f310-11e6-966a-89a5051cb4df.png">
</p>


<p>  <img src="https://cloud.githubusercontent.com/assets/21232362/22962121/8e58c2d8-f310-11e6-9956-53e4314c18fd.png">
</p>

#### Looking at last 1600 epochs
The cost function seems to be decreasing almost linearly with a slight dip. The alpha and gradient functions seem to have the same structure.










