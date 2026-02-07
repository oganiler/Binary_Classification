#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

/**
 * Binary Classification using Logistic Regression
 *
 * Mathematical Foundation:
 *
 * 1. Hypothesis Function (Sigmoid):
 *    h(x) = σ(z) = 1 / (1 + e^(-z))
 *    where z = w^T * x + b
 *
 * 2. Binary Cross-Entropy Loss:
 *    L(y, ŷ) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
 *
 *    Cost Function (average over m samples):
 *    J(w,b) = (1/m) * Σ L(y_i, ŷ_i)
 *
 * 3. Gradients (for gradient descent):
 *    ∂J/∂w = (1/m) * Σ (ŷ_i - y_i) * x_i
 *    ∂J/∂b = (1/m) * Σ (ŷ_i - y_i)
 *
 * 4. Weight Update:
 *    w = w - α * ∂J/∂w
 *    b = b - α * ∂J/∂b
 *    where α is the learning rate
 */

class LogisticRegression {
private:
    std::vector<double> weights;  // Weight vector w
    double bias;                  // Bias term b
    double learning_rate;
    int n_features;

public:
    LogisticRegression(int num_features, double lr = 0.01)
        : n_features(num_features), learning_rate(lr), bias(0.0) {

        // Initialize weights randomly with small values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, 0.01);

        weights.resize(num_features);
        for (int i = 0; i < num_features; i++) {
            weights[i] = dis(gen);
        }
    }

    /**
     * Sigmoid activation function
     *
     * Math: σ(z) = 1 / (1 + e^(-z))
     *
     * Properties:
     * - Output range: (0, 1)
     * - σ(0) = 0.5
     * - As z → ∞, σ(z) → 1
     * - As z → -∞, σ(z) → 0
     */
    double sigmoid(double z) {
        // Numerical stability: clip z to prevent overflow
        if (z > 500) return 1.0;
        if (z < -500) return 0.0;

        return 1.0 / (1.0 + std::exp(-z));
    }

    /**
     * Forward pass: compute prediction
     *
     * Math:
     *   z = w^T * x + b = Σ(w_i * x_i) + b
     *   ŷ = σ(z)
     */
    double predict_proba(const std::vector<double>& x) {
        // Compute z = w^T * x + b
        double z = bias;
        for (int i = 0; i < n_features; i++) {
            z += weights[i] * x[i];
        }

        // Apply sigmoid activation: ŷ = σ(z)
        return sigmoid(z);
    }

    /**
     * Binary prediction (0 or 1)
     * Uses threshold of 0.5
     */
    int predict(const std::vector<double>& x) {
        return predict_proba(x) >= 0.5 ? 1 : 0;
    }

    /**
     * Binary Cross-Entropy Loss for single sample
     *
     * Math: L(y, ŷ) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
     *
     * Intuition:
     * - If y=1, loss = -log(ŷ): penalizes if ŷ is far from 1
     * - If y=0, loss = -log(1-ŷ): penalizes if ŷ is far from 0
     * - Always non-negative
     */
    double binary_cross_entropy(double y_true, double y_pred) {
        // Add small epsilon to prevent log(0)
        const double epsilon = 1e-15;
        y_pred = std::max(epsilon, std::min(1.0 - epsilon, y_pred));

        return -(y_true * std::log(y_pred) + (1.0 - y_true) * std::log(1.0 - y_pred));
    }

    /**
     * Compute average loss over dataset
     *
     * Math: J(w,b) = (1/m) * Σ L(y_i, ŷ_i)
     */
    double compute_cost(const std::vector<std::vector<double>>& X,
        const std::vector<int>& y) {
        double total_loss = 0.0;
        int m = X.size();

        for (int i = 0; i < m; i++) {
            double y_pred = predict_proba(X[i]);
            total_loss += binary_cross_entropy(y[i], y_pred);
        }

        return total_loss / m;
    }

    /**
     * Train the model using Gradient Descent
     *
     * Math:
     * For each iteration:
     *   1. Compute predictions: ŷ_i = σ(w^T * x_i + b)
     *   2. Compute gradients:
     *      dw = (1/m) * Σ (ŷ_i - y_i) * x_i
     *      db = (1/m) * Σ (ŷ_i - y_i)
     *   3. Update parameters:
     *      w = w - α * dw
     *      b = b - α * db
     *
     * Derivation of gradient:
     *   ∂L/∂w = ∂L/∂ŷ * ∂ŷ/∂z * ∂z/∂w
     *   where ∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
     *         ∂ŷ/∂z = ŷ(1-ŷ)  [derivative of sigmoid]
     *         ∂z/∂w = x
     *
     *   After simplification: ∂L/∂w = (ŷ - y) * x
     */
    void train(const std::vector<std::vector<double>>& X,
        const std::vector<int>& y,
        int epochs = 1000,
        bool verbose = true) {

        int m = X.size();  // Number of training samples

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Initialize gradient accumulator
            std::vector<double> dw(n_features, 0.0);
            double db = 0.0;

            // Compute gradients by iterating through all samples
            for (int i = 0; i < m; i++) {
                // Forward pass: compute prediction
                double y_pred = predict_proba(X[i]);

                // Error term: (ŷ - y)
                double error = y_pred - y[i];

                // Accumulate gradients
                // ∂J/∂w_j += (1/m) * error * x_j
                for (int j = 0; j < n_features; j++) {
                    dw[j] += error * X[i][j];
                }

                // ∂J/∂b += (1/m) * error
                db += error;
            }

            // Average the gradients (divide by m)
            for (int j = 0; j < n_features; j++) {
                dw[j] /= m;
            }
            db /= m;

            // Update weights and bias using gradient descent
            // w = w - α * ∂J/∂w
            for (int j = 0; j < n_features; j++) {
                weights[j] -= learning_rate * dw[j];
            }

            // b = b - α * ∂J/∂b
            bias -= learning_rate * db;

            // Print progress
            if (verbose && (epoch % 100 == 0 || epoch == epochs - 1)) {
                double cost = compute_cost(X, y);
                std::cout << "Epoch " << std::setw(4) << epoch
                    << " | Cost: " << std::fixed << std::setprecision(6) << cost
                    << std::endl;
            }
        }
    }

    /**
     * Compute accuracy on dataset
     */
    double compute_accuracy(const std::vector<std::vector<double>>& X,
        const std::vector<int>& y) {
        int correct = 0;
        int m = X.size();

        for (int i = 0; i < m; i++) {
            int prediction = predict(X[i]);
            if (prediction == y[i]) {
                correct++;
            }
        }

        return static_cast<double>(correct) / m;
    }

    // Getters for inspection
    const std::vector<double>& get_weights() const { return weights; }
    double get_bias() const { return bias; }
};


/**
 * Helper function to generate synthetic data for demonstration
 * Creates a simple linearly separable dataset
 */
void generate_synthetic_data(std::vector<std::vector<double>>& X,
    std::vector<int>& y,
    int n_samples = 100) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<> dis(0.0, 1.0);

    X.clear();
    y.clear();

    for (int i = 0; i < n_samples; i++) {
        std::vector<double> sample(2);

        if (i < n_samples / 2) {
            // Class 0: centered around (-1, -1)
            sample[0] = dis(gen) - 1.5;
            sample[1] = dis(gen) - 1.5;
            y.push_back(0);
        }
        else {
            // Class 1: centered around (1, 1)
            sample[0] = dis(gen) + 1.5;
            sample[1] = dis(gen) + 1.5;
            y.push_back(1);
        }

        X.push_back(sample);
    }
}


int main() {
    std::cout << "=== Binary Classification with Logistic Regression ===" << std::endl;
    std::cout << std::endl;

    // Generate synthetic training data
    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;
    generate_synthetic_data(X_train, y_train, 200);

    std::cout << "Generated " << X_train.size() << " training samples" << std::endl;
    std::cout << "Number of features: " << X_train[0].size() << std::endl;
    std::cout << std::endl;

    // Create and train model
    LogisticRegression model(2, 0.1);  // 2 features, learning rate = 0.1

    std::cout << "Training model..." << std::endl;
    model.train(X_train, y_train, 1000, true);
    std::cout << std::endl;

    // Evaluate on training data
    double train_accuracy = model.compute_accuracy(X_train, y_train);
    std::cout << "Training Accuracy: " << std::fixed << std::setprecision(4)
        << train_accuracy * 100 << "%" << std::endl;
    std::cout << std::endl;

    // Display learned parameters
    std::cout << "Learned parameters:" << std::endl;
    std::cout << "Weights: [";
    const auto& weights = model.get_weights();
    for (size_t i = 0; i < weights.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << weights[i];
        if (i < weights.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Bias: " << std::fixed << std::setprecision(4)
        << model.get_bias() << std::endl;
    std::cout << std::endl;

    // Test predictions on a few samples
    std::cout << "Sample predictions:" << std::endl;
    for (int i = 0; i < 50; i++) {
        double prob = model.predict_proba(X_train[i]);
        int pred = model.predict(X_train[i]);
        std::cout << "Sample " << i << ": x=["
            << std::fixed << std::setprecision(2)
            << X_train[i][0] << ", " << X_train[i][1]
            << "] | P(y=1)=" << std::setprecision(4) << prob
            << " | Predicted=" << pred
            << " | Actual=" << y_train[i] << std::endl;
    }

    return 0;
}