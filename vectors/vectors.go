package vectors

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// EuclidianDistance calculates the distance between 2 vectors in space
func EuclidianDistance(a, b mat.Vector) float64 {
	var total float64
	for i := 0; i < a.Len(); i++ {
		total += math.Pow(a.AtVec(i)-b.AtVec(i), 2)
	}
	return math.Sqrt(total)
}
