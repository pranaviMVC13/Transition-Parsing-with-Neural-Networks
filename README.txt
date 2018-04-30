==========================================================================================

         Introduction to Natural Laguage Processing Assignment 3
 
==========================================================================================

To run type: python DependencyParser.py       in the terminal from the project folder.

****** Implementaion Details ******

1. ARC STANDING ALGORITHM

--This has been implemeted in 'ParsingSystem.py'
--The three cases of left arc, right arc and shit are implemeted based on the respective conditions 
( similar to the ones in can apply )
--The label is identified from the transition and the arc is drawn between the elemente fetched from stack.

------------------------------------------------------------------------------------------------------------------------------------------------

2. FEATURE EXTRACTION

--This has been implemented in 'DependencyParser.py' , getFeatures() function
--Words : Top three elements of stack and stack, left most child and right most child of these elements, left and 
right most children of previously found left and right children. getWordID() is used to get the wordID of the 
index numbers returned from the stack. This accounts to 18 words.
--Now the pos tags for these 18 words are found using getPOSID() function. This accounts to 18 pos tags
--Now the labels for all the left and right most children are found using the getLabelID() function. 
This accounts to  12 labels.
--Total we have 48 features. ( 18 words, 18 pos tags, 12 labels)

------------------------------------------------------------------------------------------------------------------------------------------------

3.NEURAL NETWORK ARCHITECHTURE

--Lookup for the embeddings for training_inputs in self.embeddings
--Randomly initialize the weight_inputs for the training inputs
--In the forward pass function , calculate the hidden unit by a mapping on weighted sum of training_inputs. 
This was calculated using the cube activation function.
--Other similar functions used in experimentation :
sigmoid : tf.sigmoid()
Tanh: tf.nn.tanh()
Relu: tf.nn.relu()

------------------------------------------------------------------------------------------------------------------------------------------------

4.LOSS FUNCTION

--A softmax layer is added on top of the hidden layer
--We are using cross entropy to calculate the loss. tf.nn.softmax_cross_entropy_with_logits_v2(). This gives the loss values.
--Also we calculate the regularization term for the parameters(weights, biases and feature_embeddings) using config.lam from config.py and tf.nn.l2_loss()
--The sum of loss and regularization term should be minimised

--------------------------------------------------------------------------------------------------------------------------------------------------

5.CONFIG.PY

--Contains the parameters like learning rate, iterations etc to experiment with.
--For the best model, the learinng rate has been maintained as 0.1 and the max_iterations is modified to 2001.
--This gave the best accuracy : 75.574
--The results_test.conll file contains the results from the above model.



