// NeuralNetCpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <random>
#include <fstream>
#include <string>
#include <sstream>

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
	return 2.0*((double)rand()) / ((double)RAND_MAX)-1.0;
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

double phi(double x) {
	return 1.0 / (1.0 + exp(-x));
}

vector<double> phi(vector<double> v) {
	vector<double> p(v.size());
	transform(v.begin(), v.end(), p.begin(), static_cast<double(*)(double)>(&phi));
	return p;
}

void FeedForward(vector<Layer*> Layers, vector<Weights*> Weightss) {
	for (int i = 1; i < Layers.size(); i++) {
		(Layers[i])->neurons = phi(((Weightss[i - 1])->w.Transpose())*((Layers[i - 1])->neurons));
	}
}

void BackPropagate(vector<Layer*> Layers, vector<Weights*> Weightss) {
	double speed = 1;
	Matrix dweights;
	int i = (Weightss.size() - 1);
	Layers[i + 1]->set_delta(*Layers[i + 1], *Weightss[i]);
	dweights = ScalarTimesMatrix(-speed, OuterProduct(Layers[i]->neurons, Layers[i + 1]->delta));
	Weightss[i]->w = Weightss[i]->w + dweights;
	for (int i = (Weightss.size() - 2); i >= 0; i--) {
		Layers[i + 1]->set_delta(*Layers[i + 2], *Weightss[i + 1]);
		dweights = ScalarTimesMatrix(-speed, OuterProduct(Layers[i]->neurons, Layers[i + 1]->delta));
		Weightss[i]->w = Weightss[i]->w + dweights;
	}
}

vector<Weights> RunNetwork(double speed, vector<double> inputs, vector<double> desired_outputs, int num_layers, int num_steps = 1, vector<Weights> Wvec = {}) {
	cout << "running...\n";
	vector<Layer> Lvec(num_layers);
	Lvec[0].neurons = inputs;
	Lvec[Lvec.size() - 1].desired = desired_outputs;
	if (Wvec.size() == 0) {
		for (int i = 0; i < num_layers - 2; i++) {
			Weights W(inputs.size(), inputs.size());
			W.w = ScalarTimesMatrix(pow(1.0 / 784.0,2.0), W.w);
			Wvec.push_back(W);
		}
		Weights W(inputs.size(), desired_outputs.size());
		Wvec.push_back(W); 
		W.w = ScalarTimesMatrix(pow(1.0 / 784.0, 2.0), W.w);
	}
	vector<Layer*>Lvec_pointers(Lvec.size());
	transform(Lvec.begin(), Lvec.end(), Lvec_pointers.begin(), [](Layer &l) {return &l; });
	vector<Weights*>Wvec_pointers(Wvec.size());
	transform(Wvec.begin(), Wvec.end(), Wvec_pointers.begin(), [](Weights &w) {return &w; });
	for (int i = 0; i < num_steps; i++) {

		FeedForward(Lvec_pointers, Wvec_pointers);
		BackPropagate(Lvec_pointers, Wvec_pointers);
	}
	cout << "Error: " << pow(Lvec[Lvec.size() - 1].neurons[0] - Lvec[Lvec.size() - 1].desired[0], 2.0) << "\n";
	return Wvec;
}

class training_data {
public:
	vector<double> labels;
	vector<vector<double>> data;

	training_data(string);
};

vector<string> getNextLineAndSplitIntoTokens(istream& str) {
	vector<string> result;
	string line;
	getline(str, line);

	stringstream lineStream(line);
	string cell;

	while (getline(lineStream, cell, ',')) {
		result.push_back(cell);
	}
	return result;
}

training_data::training_data(string filename) {
	ifstream myfile("mnist_train.csv");
	vector<string> S(1);
	S = getNextLineAndSplitIntoTokens(myfile);
	int i = 0;
	while (i < 100) {
		labels.push_back(stod(S[0],nullptr)/10.0);
		vector<double> dataline(S.size() - 1);
		transform(S.begin() + 1, S.end(), dataline.begin(), [](string s) {return stod(s, nullptr)/255.0; });
		data.push_back(dataline);
		S = getNextLineAndSplitIntoTokens(myfile);
		i++;
	}
}


int main() {
	set_seed();
	training_data train("mnist_train.csv");

	double speed = 0.1;
	int num_layers = 2;
	int num_steps = 100;
	vector<Weights> Wvec;
	Wvec = RunNetwork(speed, train.data[0], { train.labels[0] }, num_layers, num_steps);
	for (int i = 1; i < train.labels.size(); i++) {
		Wvec = RunNetwork(speed, train.data[i], { train.labels[i] }, num_layers, num_steps, Wvec);
	}
	system("pause");
	return 0;
}