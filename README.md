# Hard Hat Safety Project

## Background and Introduction

Head injuries are a common risk on construction sites. The construction industry has the highest rate of both fatal and nonfatal TBI (traumatic brain injury) across all industries in the U.S. Between 2003 and 2010, 2210 construction workers died because of a TBI, which accounted for 25% of all construction fatalities. [^1] These injuries can result in serious long-term effects including memory loss, fractured bones, spine damage, or even death for the workers involved. [^2] In addition, head injuries also cost a lot of money for the companies involved - one incident of severe TBI is estimated to cost a company between $600,000 and $1,875,000 over the worker's lifetime. [^3] The main factor behind head injuries was lack of adequate head protection: a survey by the Bureau of Labor Statistics found that 84% of workers who suffered head injuries were not wearing proper head protection at the time. [^4] This project aims to decrease the number of head injuries on construction sites by detecting whether or not the worker is wearing a helmet based on an input image. It only takes a couple seconds to classify an image, and could be used on the entryway at construction sites as a quick and easy way to verify workers wearing hard hats. 

![add image descrition here](direct image link here)

## The Algorithm

This project was developed using a Jetson Nano, imagenet.py, and a retrained resnet18 model. 

The algorithm was first trained with a series of images (from the training set) of workers wearing hard hats and workers without. The neural network was first trained on the same  data over and over, and it learned to recognize certain features of the two categories; for example, as workers wearing hard hats also wore neon safety vests more often than workers without hard hats, sp the AI learned to associate the two and later struggled with identifying workers with vests but without hard hats. 

After each passing of training data through the training data, or epoch, the trained model went through a second series of images (from the validation set) to validate its performance. This helps the model see whether its training is going in the right direction or not, as it could become an expert at classifying the data it was given but not on new data. 

When the training was complete, the model was given a seperate set of data it had never seen before (from the test set). It then had to determine if the worker was wearing a hard hat or not. The test set was used as the final performance metric used to determine how well our model performed.

## Running this project

### Setting up
1. Download *Hard_Hat_Data* from github. There should be 6 folders inside - *Test_Hat*, *Train_Hat*, *Val_Hat*, *Test_NoHat*, *Train_NoHat*, *Val_NoHat*.
2. Change directories into **jetson-inference/python/training/classification/data**.
3. Create a directory named hard_hat using `mkdir hard_hat`.
4. Move into hard_hat using `cd hard_hat`. Create directories test, train, and val using `mkdir test`, `mkdir train`, and `mkdir val`.
5. For each of test, train, and val, create two new directories inside called hard_hat and no_hard_hat.
6. Add relevant pictures from the file downloaded into the new created folders (you can use filezilla to do this step). For example, files from *Test_NoHat* should go into **jetson-inference/python/training/classification/data/hard_hat/test/no_hard_hat**. Be careful to not add the actual folders themselves, but just the image files inside.
7. Back out to **jetson-inference/python/training/classification/data/hard_hat/**, and create a new labels.txt using `cat > labels.txt`. 
8. In labels.txt, type the following 3 lines (line 3 should be left blank. Press enter to go to the next line. When you are done, exit using control-d.
    - 1 hard_hat
    - 2 no_hard_hat
    - 3
9. Move to **jetson-inference/python/training/classification/models/** and create directory hard_hat using `mkdir hard_hat`.

### Training your model

10. Move back to the **jetson-inference/** directory. run `./docker/run.sh` to enter the docker container.
11. From inside the docker container, move to **jetson-inference/python/training/classification**.
12. Run the training script: `python3 train.py --model-dir=models/hard_hat data/hard_hat`. 
13. Before you run this, you probably want to specify how many epochs you want to train the model for. do this by adding `--epochs=[epoch_number]` at the end of the previous command. For example, to train for 30 epochs, run `python3 train.py --model-dir=models/hard_hat data/hard_hat --epochs=30`.
14. Your model should be training with the data given in the data directory. It may take a minute to start. If you want to stop the training at any time, do control+c.

### Testing your model
15. When your model is trained to your heart's content, you can test it out. Make sure you are still in the docker container's **jetson-inference/python/training/classification**.
16. Run `python3 onnx_export.py --model-dir=models/hard_hat` to export your model.
17. Go to **jetson-inference/python/training/classification/models/hard_hat** to see if there is a new file called resnet18.onnx. This is your trained model.
18. Exit the docker container using ctl+d. From your nano, move to **jetson-inference/python/training/classification**.
19. Use `ls models/hard_hat/` to check that your trained model is on the nano.  A file named resnet18.onnx should be there.
20. Set the NET and DATASET variables using `NET=models/hard_hat` and `DATASET=data/hard_hat`
21. In order to see how your trained model works on an image already uploaded, use
`imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/hard_hat/05.jpg hard_hat_test_01.jpg`. You should get a test image named hard_hat_01.jpg under **jetson-inference/python/training/classification**.
    - Here, hard_hat is the current category (out of hard_hat or no_hard_hat) that the model will pull an image from to test on. You can change this to no_hard_hat if you want to test an image without a hard hat.
    - 05.jpg is the name of the image tested. You can change this to any image name you want to be tested inside hard_hat or no_hard_hat. There is 01.jpg to 20.jpg available with the downloaded package, but you can also add your own images into the folder and input its name to test it.
    - hard_hat_01.jpg is the name that the tested image will be saved as. Change this to what you want the output name to be.
22. If you want to run all of the test images at once, follow these steps:

    - Create two new directories using `mkdir $DATASET/test_output_hard_hat $DATASET/test_output_no_hard_hat`. Run these commands:
    - `imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/hard_hat $DATASET/test_output_hard_hat`
    - `imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test/no_hard_hat $DATASET/test_output_no_hard_hat`

Now, in **jetson-inference/python/training/classification/data/hard_hat**, you should see two new directories named test_output_hard_hat and test_ouput_no_hard_hat. Inside, you should see all your tested images and their results.

## Extra

When you download the Hard_Hat_Safety-main folder from github, click inside Hard_Hat_Data to see a .py file called trainedmodel.py. This is a model that has been trained for about 820 epochs. With the val images, it was able to detect about 88-92% of the val images correctly. It also had 20% of false positives (correctly identified 80% of validation images that had workers with a construction hat) and 0% false negatives (correctly identified 100% of validation images that had workers without a construction site). This was a positive result because if workers are wearing hats and the model detects otherwise, workers can walk through, and there are less chances of a worker without proper head protection passing through as the model can detect their wearing a hard hat.

[View a video explanation here](video link)

[^1]: https://blogs.cdc.gov/niosh-science-blog/2022/11/10/construction-helmets/
[^2]: https://www.brainandspinalcord.org/cost-traumatic-brain-injury/
[^3]: https://www.cdc.gov/traumaticbraininjury/moderate-severe/potential-effects.html
[^4]: https://ohsonline.com/Articles/2021/06/01/Watch-for-Falling-Objects-PPE-to-Protect-You-on-the-Job-Site.aspx
