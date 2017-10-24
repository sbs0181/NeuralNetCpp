// NeuralNetCpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <random>

using namespace std;

template<class T>
void PrintVector(vector<T> v) {
	for (int i = 0; i < v.size(); i++) {
		cout << v[i] << "\t";
	}
	cout << endl;
}

vector<double> ScalarTimesVector(double s, vector<double>v) {
	vector<double> v2(v.size());
	transform(v.begin(), v.end(), v2.begin(), [&s](double d) {return s*d; });
	return v2;
}

vector<double> VectorTimesVector(vector<double> v1, vector<double> v2) {
	vector<double> v3(v1.size());
	transform(v1.begin(), v1.end(), v2.begin(), v3.begin(), multiplies<double>());
	return v3;
}

vector<double> VectorPlusVector(vector<double> v1, vector<double> v2) {
	vector<double> v3(v1.size());
	transform(v1.begin(), v1.end(), v2.begin(), v3.begin(), plus<double>());
	return v3;
}

vector<double> VectorMinusVector(vector<double> v1, vector<double> v2) {
	vector<double> v3(v1.size());
	transform(v1.begin(), v1.end(), v2.begin(), v3.begin(), minus<double>());
	return v3;
}

vector<double> ScalarPlusVector(double d, vector<double>v) {
	vector<double> v2(v.size());
	transform(v.begin(), v.end(), v2.begin(), [&d](double s) {return s + d; });
	return v2;
}




class Matrix {
public:
	vector<vector<double>> m;

	vector<double> get_row(int);
	vector<double> get_col(int);
	Matrix Transpose();
	Matrix operator*(Matrix &B);
	vector<double> operator*(vector<double> &v);
	Matrix operator+(Matrix &B);
	
};

vector<double> Matrix::get_row(int i) {
	return m[i];
}

vector<double> Matrix::get_col(int i) {
	vector<double> vec;
	for (int j = 0; j < m.size(); j++) {
		vec.push_back(m[j][i]);
	}
	return vec;
}

Matrix Matrix::Transpose() {
	Matrix T;
	for (int i = 0; i < m[0].size(); i++) {
		T.m.push_back(get_col(i));
	}
	return T;
}

vector<double> Matrix::operator*(vector<double> &B) {
	vector<double> vec;
	vector<double> tvec(B.size());
	vector<double> t;
	for (int i = 0; i < m.size(); i++) {
		t = get_row(i);
		transform(t.begin(), t.end(), B.begin(), tvec.begin(), [](double m, double v) {return m*v; });
		vec.push_back(accumulate(tvec.begin(), tvec.end(), 0.0));
	}
	return vec;
}

Matrix Matrix::operator*(Matrix &B) {
	Matrix C;
	for (int i = 0; i < B.m[0].size(); i++) {
		C.m.push_back((*this)*(B.get_col(i)));
	}
	return C.Transpose();
}

Matrix Matrix::operator+(Matrix &B) {
	Matrix C;
	vector<double> t(B.m[0].size());
	for (int i = 0; i < B.m.size(); i++) {
		transform(m[i].begin(), m[i].end(), B.m[i].begin(), t.begin(), [](double d1, double d2) {return d1 + d2; });
		C.m.push_back(t);
	}
	return C;
}

Matrix ScalarTimesMatrix(double d,Matrix B) {
	Matrix C = B;
	for (int i = 0; i < C.m.size(); i++) {
		transform(B.m[i].begin(), B.m[i].end(), C.m[i].begin(), [&d](double x) {return x*d; });
	}
	return C;
}

Matrix OuterProduct(vector<double> v1, vector<double> v2) {
	Matrix C;
	for (int i = 0; i < v1.size(); i++) {
		C.m.push_back(ScalarTimesVector(v1[i],v2));
	}
	return C;
}

void set_seed() {
	srand(time(NULL));
}

double unifrnd() {
	return ((double)rand()) / ((double)RAND_MAX);
}

vector<double> unifrnd(int n) {
	vector<double> v;
	for (int i = 0; i < n; i++) {
		v.push_back(unifrnd());
	}
	return v;
}

vector<vector<double>> unifrnd(int n,int m) {
	vector<vector<double>> v;
	for (int i = 0; i < m; i++) {
		v.push_back(unifrnd(n));
	}
	return v;
}

class Weights {
public:
	Matrix w;
	Weights(int,int);
	Weights() {};
};

Weights::Weights(int n_inputs, int n_outputs) {
	w.m = unifrnd(n_outputs, n_inputs);
}

class Layer {
public:
	vector<double> neurons;
	
	vector<double> desired;

	vector<double> delta;
	void set_delta(Layer L, Weights W);
	Layer() {};
};

void Layer::set_delta(Layer L, Weights W) {
	if (desired.size() == 0) {
		Matrix M = W.w;
		vector<double> v = L.delta;
		vector<double> mv = M*v;
		delta = VectorTimesVector(VectorTimesVector(W.w*L.delta, neurons), ScalarPlusVector(1.0, ScalarTimesVector(-1.0, neurons)));
	}
	else {
		delta = VectorTimesVector(VectorTimesVector(VectorMinusVector(neurons, desired), neurons), ScalarPlusVector(1.0, ScalarTimesVector(-1.0, neurons)));
	}
}

void FeedForward(vector<Layer*> Layers, vector<Weights*> Weightss) {
	for (int i = 1; i < Layers.size(); i++) {
		(Layers[i])->neurons = ((Weightss[i - 1])->w.Transpose())*((Layers[i - 1])->neurons);
	}
}

void BackPropagate(vector<Layer*> Layers, vector<Weights*> Weightss) {
	double speed = 0.1;
	Matrix dweights;
	int i = (Weightss.size() - 1);
	Layers[i + 1]->set_delta(*Layers[i + 1], *Weightss[i]);
	dweights = ScalarTimesMatrix(-speed, OuterProduct(Layers[i]->neurons, Layers[i + 1]->delta));
	Weightss[i]->w = Weightss[i]->w + dweights;
	for (int i = (Weightss.size()-2); i >=0; i--) {
		Layers[i + 1]->set_delta(*Layers[i + 2], *Weightss[i+1]);
		dweights = ScalarTimesMatrix(-speed, OuterProduct(Layers[i]->neurons, Layers[i + 1]->delta));
		Weightss[i]->w = Weightss[i]->w + dweights;
	}
}



int main() {
	set_seed();
	double speed = 0.1;
	Layer L1, L2, L3, L4;
	L1.neurons = ScalarTimesVector(0.1,unifrnd(4));
	L4.desired = unifrnd(2);
	Weights W1(4, 4);
	Weights W2(4, 4);
	Weights W3(4, 2);
	//L2.neurons = W.w.Transpose()*L1.neurons;
	vector<Layer*> Lvec = { &L1,&L2,&L3,&L4 };
	vector<Weights*> Wvec{ &W1,&W2,&W3 };
	FeedForward(Lvec, Wvec);
	PrintVector(L4.neurons);
	
	//Matrix dweights=ScalarTimesMatrix(-speed,OuterProduct(L1.neurons,L2.delta()));
	//W.w = W.w + dweights;
	//PrintVector((dweights).m[0]);
	for (int i = 0; i < 1000; i++) {
		BackPropagate(Lvec, Wvec);
		FeedForward(Lvec, Wvec);
		PrintVector(L4.neurons);
	}
	PrintVector(L4.desired);
	system("pause");
	return 0;
}







/*
double phi(double x) { return 1.0 / (1.0 + exp(-x)); }

class Neuron {
public:
	vector<double> inputweights;
	vector<double> outputweights;

	vector<double> inputvalues;
	vector<double> outputdeltas;

	double value;
	double speed;
	double delta;

	void SetInputValues(vector<Neuron>);
	void SetOutputDeltas(vector<Neuron>);

	void Forward();
	void Backward();
};

void Neuron::SetInputValues(vector<Neuron> nvec) {
	vector<double> iv(nvec.size());
	transform(nvec.begin(), nvec.end(), iv.begin(), [](Neuron n) {return n.value; });
	inputvalues = iv;
}

void Neuron::SetOutputDeltas(vector<Neuron> nvec) {
	vector<double> od(nvec.size());
	transform(nvec.begin(), nvec.end(), od.begin(), [](Neuron n) {return n.delta; });
	outputdeltas = od;
}

void Neuron::Forward() {
	vector<double> newvec(inputweights.size());
	transform(inputweights.begin(), inputweights.end(), inputvalues.begin(), newvec.begin(), [](double w, double v) {return w*v; });
	value = phi(accumulate(newvec.begin(), newvec.end(), 0.0));
}

void Neuron::Backward() {
	vector<double> newweights(inputweights.size());
	transform(inputweights.begin(), inputweights.end(), inputvalues.begin(), newweights.begin(), [this](double w, double v) {return w - (this->speed)*(this->delta)*v; });
	inputweights = newweights;
}

int main() {
	Neuron n1, n2, n3;
	n1.value = 1.0;
	n2.value = 0.5;
	n3.SetInputValues({ n1,n2 });
	n3.inputweights={}


	return 0;
}
*/
/*
class Neuron {
public:
	vector<double> weights;

	double value;
	double speed;

	double delta() {};
	
	void evaluate(vector<Neuron>);
	void UpdateWeights(vector<Neuron>);
};

class InputNeuron : public Neuron {
public:
	void evaluate(vector<Neuron>) {};
	void evaluate(double d) { value = d; };

	double delta() {};

	void UpdateWeights(vector<Neuron>) {};
};

class OutputNeuron : public Neuron {
public:
	double desired;
	double delta() {
		return (value - desired)*value*(1.0 - value);
	};
	

};

void Neuron::evaluate(vector<Neuron> nvec) {
	if (nvec.size() != weights.size()) {
		_DEBUG_ERROR("Weights and input not equal size");
	}
	vector<double> vec(weights.size());
	transform(nvec.begin(), nvec.end(), weights.begin(), vec.begin(), [](Neuron n, double d) {return d*(n.value); });
	value = phi(accumulate(vec.begin(),vec.end(),0.0));

}

void Neuron::UpdateWeights(vector<Neuron> nvec) {
	vector<double> newweights(nvec.size());
	transform(weights.begin(), weights.end(), nvec.begin(), newweights.begin(), [this](double d, Neuron n) {return d - (this->speed)*(this->delta())*n.value; });
}


int main()
{
	InputNeuron n1;
	InputNeuron n2;
	OutputNeuron n3;
	n3.weights = { 0.1,0.1 };

	n1.evaluate(0.1);
	n2.evaluate(0.2);
	n3.evaluate({ n1,n2 });
	cout << n3.value << endl;
	system("pause");


    return 0;
}
*/
