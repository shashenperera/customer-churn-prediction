# Dataset

This project uses the **Streaming Service Data** dataset obtained from Kaggle.

The dataset contains **5,000 customer records and 12 features**, covering demographic, subscription, payment, support, satisfaction, activity, and spending information.

The target variable used for customer churn prediction is:

```         
Churned
```

## Source

**Kaggle:** [Streaming Service Data](https://www.kaggle.com/datasets/akashanandt/streaming-service-data)

The dataset is used in this project for educational and portfolio purposes.

## License

The dataset is released under the **MIT License**.

The original license notice and copyright attribution are retained below:

```         
MIT License  Copyright (c) 2013 Mark Otto. Copyright (c) 2017 Andrew Fong.  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Local Setup

The notebook expects the dataset to be available at:

```         
data/churn.csv
```

If the dataset is included in the repository, place the CSV file in the `data/` directory using the filename:

```         
churn.csv
```

The notebook loads the dataset using a relative path:

```         
pd.read_csv("data/churn.csv")
```

This allows the project to be run on another computer without relying on machine-specific file paths.
