package vectors

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestEuclidianDistance(t *testing.T) {
	testCases := []struct {
		A        mat.Vector
		B        mat.Vector
		Distance float64
	}{
		{
			A:        mat.NewVecDense(3, []float64{1, 1, 1}),
			B:        mat.NewVecDense(3, []float64{1, 1, 1}),
			Distance: 0,
		},
		// TODO add more test cases here
	}

	for _, c := range testCases {
		d := EuclidianDistance(c.A, c.B)
		if d != c.Distance {
			t.Errorf("Expected %f, got %f", c.Distance, d)
		}
	}
}
