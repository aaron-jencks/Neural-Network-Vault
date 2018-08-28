using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace Genetic_Algorithm_Toolkit
{
    /// <summary>
    /// A parent class for a generic genetic algorithm
    /// it impliments the IGA interface
    /// 
    /// @author Aaron Jencks
    /// </summary>
    public abstract class GA : IGA, IObservable<TrainingUpdateEventArgs>
    {
        #region Properties

        #region interface properties

        public ICollection<ICitizen> Population { get; set; }

        #endregion

        /// <summary>
        /// Boolean indicator determining whether mutation should occur after crossover
        /// </summary>
        public bool DoMutation { get; set; } = true;

        /// <summary>
        /// The percent chance (0-1) indicating how often that citizens are mutated
        /// </summary>
        public double MutationChance { get; set; } = 0.0001;

        /// <summary>
        /// This is the default citizen object that is used for population generation
        /// </summary>
        public Citizen defaultCitizenTemplate { get; set; }

        private List<IObserver<TrainingUpdateEventArgs>> Observers { get; set; }

        #endregion

        public GA(Citizen citizenTemplate)
        {
            Population = new List<ICitizen>();
            defaultCitizenTemplate = citizenTemplate;
            Observers = new List<IObserver<TrainingUpdateEventArgs>>();
        }

        #region Methods

        #region interface methods

        public virtual ICollection<ICitizen> GeneratePopulation(Citizen CitizenTemplate, int count)
        {
            Population = new List<ICitizen>(count);
            for (int i = 0; i < count; i++)
                Population.Add(CitizenTemplate.GenerateNew());
            return Population;
        }

        public virtual ICollection<ICitizen> Crossover()
        {
            throw new NotImplementedException();
        }

        public virtual ICollection<ICitizen> Mutation()
        {
            if(DoMutation)
            {
                Random rng = new Random();
                List<Task> threads = new List<Task>(Population.Count);
                bool isComplete = false;

                foreach(ICitizen c in Population)
                {
                    if (rng.NextDouble() <= MutationChance)
                        threads.Add(Task.Factory.StartNew(c.Mutate));    // Launches in a new thread to maximize processing speed
                }

                // Waits until all of the tasks have completed
                do
                {
                    isComplete = true;

                    foreach (Task t in threads)
                        if (!t.IsCompleted)
                        {
                            isComplete = false;
                            break;
                        }

                } while (!isComplete);
            }

            return Population;
        }

        public virtual void Selection()
        {
            List<Task> threads = new List<Task>(Population.Count);
            bool isComplete = false;

            foreach (ICitizen c in Population)
                threads.Add(Task.Factory.StartNew(c.Select));    // Launches in a new thread to maximize processing speed

            // Waits until all of the tasks have completed
            do
            {
                isComplete = true;
                foreach (Task t in threads)
                    if (!t.IsCompleted)
                    {
                        isComplete = false;
                        break;
                    }
            } while (!isComplete);
        }

        #endregion

        /// <summary>
        /// The training method called to run the course of the algorithm (ie. Generate Population, selection, crossover, mutation, repeat)
        /// </summary>
        /// <param name="iterations">Total number of iterations of populations to complete</param>
        /// <param name="populationSize">Population size</param>
        /// <param name="citizenTemplate">The template citizen to use for generation</param>
        /// <returns></returns>
        public CancellationTokenSource Train(int iterations, int populationSize = 100, Citizen citizenTemplate = null)
        {
            GeneratePopulation(citizenTemplate ?? defaultCitizenTemplate, populationSize);  // Generates the initial Population

            // Handles the abortion of the training procedure if ended prematurely
            CancellationTokenSource TokenSource = new CancellationTokenSource();
            CancellationToken token = TokenSource.Token;

            // Launches the training task in a new thread, that is cancellable
            Task.Factory.StartNew(() =>
            {
                token.ThrowIfCancellationRequested();   // Tests if the task has already been cancelled

                // Handles the cancellation event asynchronously
                Task.Factory.StartNew(() =>
                {
                    while (!token.IsCancellationRequested)
                    {
                        token.ThrowIfCancellationRequested();
                        Thread.Sleep(100);
                    }
                });

                TrainingRoutine(iterations);    // Executes the training routine

            }, token);

            return TokenSource;
        }

        /// <summary>
        /// The actual method that is called by the Train method, this is launched asynchronously by the Train method, edit this instead of the Train method
        /// </summary>
        /// <param name="iterations">Total number of populations to iterate through</param>
        protected virtual void TrainingRoutine(int iterations)
        {
            for (int i = 0; i < iterations; i++)
            {
                Selection();

                Crossover();

                Mutation();

                // Signals end of a training iteration
                OnTrainingUpdateEvent(i, Population);
            }

            // Signals end of the trainig cycles
            OnTrainingFinishEvent();
        }

        #endregion

        #region Events

        #region Training Update

        public delegate void TrainingUpdateEventHandler(object sender, TrainingUpdateEventArgs e);

        /// <summary>
        /// Triggered every time that a training iteration finishes
        /// </summary>
        public event TrainingUpdateEventHandler TrainingUpdateEvent;

        /// <summary>
        /// Causes the TrainingUpdateEvent to trigger
        /// </summary>
        /// <param name="iteration">Current test iteration</param>
        /// <param name="population">Current population</param>
        protected virtual void OnTrainingUpdateEvent(int iteration, ICollection<ICitizen> population)
        {
            TrainingUpdateEvent?.Invoke(this, new TrainingUpdateEventArgs(iteration, population));
        }

        #endregion

        #region Training Complete

        public delegate void TrainingFinishEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Triggered every time that a training finishes
        /// </summary>
        public event TrainingFinishEventHandler TrainingFinishEvent;

        /// <summary>
        /// Causes the TrainingFinishEvent to trigger
        /// </summary>
        protected virtual void OnTrainingFinishEvent()
        {
            TrainingFinishEvent?.Invoke(this, new EventArgs());
        }

        #endregion

        public IDisposable Subscribe(IObserver<TrainingUpdateEventArgs> observer)
        {
            Observers.Add(observer);

            throw new NotImplementedException();
        }

        #endregion
    }
}
