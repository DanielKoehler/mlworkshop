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
	evaulateKNN()
	getBaseline()
}

func evaulateKNN() {
	var topK int
	var topScore float64

	for i := 1; i < MaxK; i++ {
		model := NewKNNClassifier(i)

		result, err := harness.Evaluate("diabetes.csv", model)
		if err != nil {
			log.Fatal(err)
		}

		log.Printf("Result (K:%d) = %f", i, result)
		if result.F1 > topScore {
			topScore = result.F1
			topK = i
		}
	}

	log.Printf("Top K: %d, ", topK)
	log.Printf("Top Score: %f\n", topScore)
}

// KNNClassifier implements the harness.Predictor interface
type KNNClassifier struct {
	k            int
	trainingData *mat.Dense
	rows         int
	labels       []string
}

func NewKNNClassifier(k int) *KNNClassifier {
	return &KNNClassifier{
		k: k,
	}
}

// Fit is dumb, do everything in predict
func (k *KNNClassifier) Fit(trainingData *mat.Dense, labels []string) {
	k.rows = len(labels)
	k.trainingData = trainingData
	k.labels = labels
}

// Predict on the test data based upon the training data
func (k *KNNClassifier) Predict(testData *mat.Dense) []string {
	lenTestData, _ := testData.Dims()
	predictions := make([]string, lenTestData)

	// loop over the test data and calculate the distance
	for i := 0; i < lenTestData; i++ {
		predictions[i] = k.makePrediction(testData.RowView(i))
	}
	// log.Println(predictions)
	return predictions
}

func (k *KNNClassifier) makePrediction(row mat.Vector) string {

	distances := make([]float64, k.rows)
	inds := make([]int, k.rows)
	for j := 0; j < k.rows; j++ {
		distances[j] = vectors.EuclidianDistance(row, k.trainingData.RowView(j))
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

	if count1 >= count0 {
		return "1"
	} else {
		return "0"
	}

}

func getBaseline() {
	model := &BaselineClassifier{}

	result, err := harness.Evaluate("diabetes.csv", model)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Baseline Result = %f", result.F1)
}

// BaselineClassifier can calulate a baseline if you set everything to 1
type BaselineClassifier struct{}

// Fit doesn't need to do anything
func (b *BaselineClassifier) Fit(trainingData *mat.Dense, labels []string) {
}

// Predict on the test data based upon the training data
func (b *BaselineClassifier) Predict(testData *mat.Dense) []string {
	rowCount, _ := testData.Dims()
	predictions := make([]string, rowCount)
	for i := 0; i < rowCount; i++ {
		predictions[i] = "1"
	}

	return predictions
}
