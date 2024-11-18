import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
# %matplotlib inline


df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

### Data Exploration
# Let's first have a descriptive exploration on our data.

# summarize the data
df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# We can plot each of these features:
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Now, let's plot each of these features against the Emission, to see how linear their relationship is:
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='teal')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()


# Let's split our dataset into train and test sets. 80% of the entire dataset will be used for training and 20% for testing. We create a mask to select random rows using np.random.rand() function:
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Simple Regression Model
# Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the actual value y in the dataset, and the predicted value yhat using linear approximation.
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Using sklearn package to model data.
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# Plot outputs
# We can plot the fit line over the data:
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#
# Evaluation
# We compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
#
# There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:
#
# Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
#
# Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean Absolute Error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
#
# Root Mean Squared Error (RMSE).
#
# R-squared is not an error, but rather a popular metric to measure the performance of your regression model. It represents how close the data points are to the fitted regression line. The higher the R-squared value, the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )



from PIL import Image
import os


def convert_to_jpg(input_file):
    try:
        # Open the image file
        img = Image.open(input_file)

        # Define output file name with .jpg extension
        output_file = os.path.splitext(input_file)[0] + '.jpg'

        # Convert and save as .jpg
        img.convert('RGB').save(output_file, 'JPEG')
        print(f"Successfully converted to {output_file}")

    except Exception as e:
        print(f"Error converting file: {e}")


# Specify your input file path
input_file = 'C:/Users/nikhilp/PycharmProjects/master/cctv_images/68--2.jpg'
convert_to_jpg(input_file)

import imghdr
from PIL import Image


def repair_image(file_path):
    try:
        # Check if the file has a valid image format
        image_type = imghdr.what(file_path)
        if image_type is None:
            print(f"File {file_path} is not a valid image or is corrupted.")
            return

        print(f"Detected image type: {image_type}")

        # Try to open the file using PIL
        img = Image.open(file_path)

        # Convert and save the image as JPG
        output_file = file_path.replace('.jpg', '_repaired.jpg')
        img.convert('RGB').save(output_file, 'JPEG')
        print(f"Successfully repaired and saved as {output_file}")

    except Exception as e:
        print(f"Error repairing image: {e}")


# Specify your file path
file_path = 'C:/Users/nikhilp/PycharmProjects/master/cctv_images/68--2.jpg'
repair_image(file_path)


def rewrite_image_as_binary(input_file, output_file):
    try:
        with open(input_file, 'rb') as file:
            binary_data = file.read()

        # Write the binary data to a new file to try and recover any usable content
        with open(output_file, 'wb') as new_file:
            new_file.write(binary_data)

        print(f"Rewritten binary data to {output_file}.")

    except Exception as e:
        print(f"Error during binary rewriting: {e}")


# Specify your file paths
input_file = 'C:/Users/nikhilp/PycharmProjects/master/cctv_images/68--2.jpg'
output_file = 'C:/Users/nikhilp/PycharmProjects/master/cctv_images/68--2_rewritten.jpg'

rewrite_image_as_binary(input_file, output_file)
