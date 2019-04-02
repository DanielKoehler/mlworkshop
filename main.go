package main

import (
	"log"

	"github.com/augier/mlworkshop/vectors"
	"github.com/fresh8/mlworkshop/harness"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

const MaxK = 100

func main() {
	var topK int
	var topScore float64

	for i := 1; i < MaxK; i++ {
		model := NewKNNClassifier(i)

		result, err := harness.Evaluate("diabetes.csv", model)
		if err != nil {
			log.Fatal(err)
		}

		log.Printf("Result (K:%d) = %f", i, result)
		if result > topScore {
			topScore = result
			topK = i
		}
	}

	log.Printf("Top K: %d, ", topK)
	log.Printf("Top Score: %f\n", topScore)
}

// KNNClassifier implements the harness.Predictor interface
type KNNClassifier struct {
	k            int
	trainingData mat.Matrix
	rows         int
	labels       []string
}

func NewKNNClassifier(k int) *KNNClassifier {
	return &KNNClassifier{
		k: k,
	}
}

// Fit is dumb, do everything in predict
func (k *KNNClassifier) Fit(trainingData mat.Matrix, labels []string) {
	k.rows = len(labels)
	k.trainingData = trainingData
	k.labels = labels

}

// Predict on the test data based upon the training data
func (k *KNNClassifier) Predict(testData mat.Matrix) []string {
	lenTestData, _ := testData.Dims()
	predictions := make([]string, lenTestData)

	// loop over the test data and calculate the distance
	for i := 0; i < lenTestData; i++ {
		distances := make([]float64, k.rows)
		inds := make([]int, k.rows)
		for j := 0; j < k.rows; j++ {
			distances[j] = vectors.EuclidianDistance(testData.(mat.RowViewer).RowView(i), k.trainingData.(mat.RowViewer).RowView(j))
		}

		floats.Argsort(distances, inds)

		smallest := inds[0:k.k]
		// log.Println(smallest)
		// log.Println(k.k)
		var count0, count1 int
		for _, x := range smallest {
			// log.Println(x)
			// log.Println(k.labels[x])
			switch k.labels[x] {
			case "1":
				count1++
			case "0":
				count0++
			}
		}

		if count1 > count0 {
			predictions[i] = "1"
		} else {
			predictions[i] = "0"
		}
	}

	// log.Println(predictions)
	return predictions
}
