use std::collections::VecDeque;
use std::io::{self, Write};
use std::usize;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::module::Module;
use burn::tensor::backend::{Backend, AutodiffBackend};
use burn::tensor::Tensor;
use burn::optim::{SgdConfig, Optimizer, GradientsParams};
use burn::nn::loss::{CrossEntropyLossConfig, MseLoss};
use burn::prelude::ToElement;

const FACTOR: f64 = 0.8;

const PATTERNS: &'static[(f64, &'static[u8])] = &[
    (0.99, &[1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (0.99, &[9,8,7,6,5,4,3,2]),
    (0.6, &[1]),
    (0.6, &[1, 2]),
    (0.6, &[2]),
    (0.6, &[2, 3]),
    (0.6, &[3]),
    (0.6, &[3, 4]),
    (0.6, &[4]),
    (0.6, &[4, 5]),
    (0.6, &[5]),
    (0.6, &[5, 6]),
    (0.6, &[6]),
    (0.6, &[6, 7]),
    (0.6, &[7]),
    (0.6, &[8]),
    (0.6, &[9]),
];

fn main() {
    const SZ: usize = 10;
    let mut answers = VecDeque::from([0u8; SZ]);
    let mut nums : Vec<u8>= Vec::with_capacity(SZ);
    let dev = NdArrayDevice::default();

    // We need a backend that supports Autodiff for the .learn() method
    type MyBackend = burn::backend::Autodiff<NdArray>;
    let mut net = ThreeLayerMlp::<MyBackend>::new(&dev);
    let mut learn_rate_inc = LearnRateInc::new(LearningMethod::Flat, 0.99);
    // let mut learn_rate_inc = LearnRateInc::new(LearningMethod::Flat, 0.999);

    // Initialize a simple SGD optimizer
    let mut optim = SgdConfig::new().init();

    let mut iteration = 0;
    let mut pattern = 0;
    let mut average_score = 0.0;
    const ALPHA:f64 = 0.1; // Adjust this: 0.1 means a "memory" of roughly the last 10-20 guesses.
    loop {
        iteration = usize::wrapping_add(iteration, 1);
        let my_number = read_guess(&mut iteration, pattern);

        // Predict based on current state (before pushing the new number)
        let guess = net.predict(&answers, &dev);
        nums.clear();
        nums.extend(answers.iter().rev());
        let mut points = 0.0;
        let emoji = match guess==my_number{
            true => {
                points = 1.0;
                "✅"
            },
            false => "❌",
        };
        average_score = (points * ALPHA) + (average_score * (1.0 - ALPHA));
        let display_percent = average_score * 100.0;
        println!("[{iteration:010} {display_percent:.2}%] My prediction was: {guess}{emoji}. You entered: {my_number}; Learning rate={:.3}, Previous = {nums:?}", FACTOR * learn_rate_inc.poll_rate_inverse());

        let desired_score = PATTERNS[pattern].0;
        if average_score >= desired_score {
            average_score = 0.0;
            pattern = (pattern + 1) % PATTERNS.len();
        }

        // Online learning: Train the network on the new data point
        // In a real scenario, you've just seen 'my_number' as the result of the previous 10.
        net.learn(&answers, my_number, &mut optim, &dev, &mut learn_rate_inc);

        // Update the history
        answers.pop_front();
        answers.push_back(my_number);
    }
}

fn read_guess(iteration: &mut usize, pattern: usize) -> u8 {
    const SIMULATE: bool = true;
    const MAX_ITERATIONS: usize = 30_000;
    let simulate = SIMULATE && *iteration<MAX_ITERATIONS;
    match simulate {
        true => {
            let (_, pattern) = PATTERNS[pattern];
            pattern[*iteration % pattern.len()]
        }
        false => {
            print!("Enter a number (1-10): ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            input.trim().parse().unwrap_or(1).clamp(1, 9)
        }
    }
}

#[derive(Module, Debug)]
pub struct ThreeLayerMlp<B: Backend> {
    input_layer: Linear<B>,
    hidden_layer_one: Linear<B>,
    hidden_layer_two: Linear<B>,
    hidden_layer_three: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
}

const MIDDLE_SZ: usize = 32;

impl<B: Backend> ThreeLayerMlp<B> {
    pub fn new(device: &B::Device) -> Self {
        let input_layer = LinearConfig::new(10, MIDDLE_SZ).init(device);
        let hidden_layer_one = LinearConfig::new(MIDDLE_SZ, MIDDLE_SZ).init(device);
        let hidden_layer_two = LinearConfig::new(MIDDLE_SZ, MIDDLE_SZ).init(device);
        let hidden_layer_three = LinearConfig::new(MIDDLE_SZ, MIDDLE_SZ).init(device);
        let output_layer = LinearConfig::new(MIDDLE_SZ, 9).init(device);

        Self {
            input_layer,
            hidden_layer_one,
            hidden_layer_two,
            hidden_layer_three,
            output_layer,
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input_layer.forward(input);
        let x = self.activation.forward(x);
        let x = self.hidden_layer_one.forward(x);
        let x = self.activation.forward(x);
        let x = self.hidden_layer_two.forward(x);
        let x = self.activation.forward(x);
        let x = self.hidden_layer_three.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn predict(&self, input: &VecDeque<u8>, device: &B::Device) -> u8 {
        let data: Vec<f32> = input.iter().map(|&v| v as f32).collect();
        let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 10]);

        let output = self.forward(tensor);

        // Change: Use argmax to find the index (0-8) with the highest probability
        let prediction_index = output.argmax(1).into_scalar().to_f32() as u8;

        // Map index 0-8 back to numbers 1-9
        prediction_index + 1
    }
}

// Extension for backpropagation
// 1. Remove the explicit Gradients=GradientsParams bound
impl<B: AutodiffBackend> ThreeLayerMlp<B> {
    pub fn learn(
        &mut self,
        input: &VecDeque<u8>,
        expected: u8,
        optim: &mut impl Optimizer<Self, B>,
        device: &B::Device,
        learn_rate_inc: &mut LearnRateInc
    ) {
        let data: Vec<f32> = input.iter().map(|&v| v as f32).collect();
        let x = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 10]);

        // Change: CrossEntropy expects class indices as Int Tensors
        // Target index is expected - 1 (to map 1-9 into 0-8)
        let y = Tensor::<B, 1, burn::tensor::Int>::from_ints([(expected - 1) as i32], device);

        let output = self.forward(x);

        // Change: CrossEntropyLoss is better for probability distributions
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output, y);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, self);

        *self = optim.step(learn_rate_inc.rate_inverse() * FACTOR, self.clone(), grads);
    }
}

#[derive(Debug)]
struct LearnRateInc {
    pub iteration: f32,
    pub rate: f32,
    pub method: LearningMethod,
}

impl LearnRateInc {
    pub fn new(method: LearningMethod, rate: f32) -> Self {
        LearnRateInc { method, rate, iteration: 1.0 }
    }

    pub fn poll_rate(&self) -> f64 {
        match self.method {
            LearningMethod::XByX => {
                (self.iteration/(self.iteration + self.rate)) as f64
            }
            LearningMethod::Flat => {
                self.rate as f64
            }
        }
    }

    pub fn poll_rate_inverse(&self) -> f64 {
        1.0 - self.poll_rate()
    }

    pub fn rate(&mut self) -> f64 {
        let r= self.poll_rate();
        self.iteration+=1.0;
        r
    }

    fn rate_inverse(&mut self) -> f64 {
        1.0 - self.rate()
    }
}

#[derive(Debug)]
enum LearningMethod {
    XByX,
    Flat,
}
