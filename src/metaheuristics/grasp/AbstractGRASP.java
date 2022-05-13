/**
 * 
 */
package metaheuristics.grasp;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Random;

import problems.Evaluator;
import solutions.Solution;

/**
 * Abstract class for metaheuristic GRASP (Greedy Randomized Adaptive Search
 * Procedure). It consider a minimization problem.
 * 
 * @author ccavellucci, fusberti
 * @param <E>
 *            Generic type of the element which composes the solution.
 */
public abstract class AbstractGRASP<E> {

	/**
	 * flag that indicates whether the code should print more information on
	 * screen
	 */
	public static boolean verbose = true;

	/**
	 * a random number generator
	 */
	static Random rng = new Random(0);

	/**
	 * the objective function being optimized
	 */
	protected Evaluator<E> ObjFunction;

	/**
	 * the GRASP greediness-randomness parameter
	 */
	protected Double alpha;

	/**
	 * the best (incumbent) solution cost
	 */
	protected Double bestCost;

	/**
	 * the current solution cost
	 */
	protected Double cost;

	/**
	 * the best solution
	 */
	protected Solution<E> bestSol;

	/**
	 * the current solution
	 */
	protected Solution<E> sol;

	/**
	 * the number of iterations the GRASP main loop executes.
	 */
	protected Integer iterations;

	/**
	 * the Candidate List of elements to enter the solution.
	 */
	protected ArrayList<E> CL;

	/**
	 * the Restricted Candidate List of elements to enter the solution.
	 */
	protected ArrayList<E> RCL;

	/**
	 * Creates the Candidate List, which is an ArrayList of candidate elements
	 * that can enter a solution.
	 * 
	 * @return The Candidate List.
	 */
	public abstract ArrayList<E> makeCL();

	/**
	 * Creates the Restricted Candidate List, which is an ArrayList of the best
	 * candidate elements that can enter a solution. The best candidates are
	 * defined through a quality threshold, delimited by the GRASP
	 * {@link #alpha} greedyness-randomness parameter.
	 * 
	 * @return The Restricted Candidate List.
	 */
	public abstract ArrayList<E> makeRCL();

	/**
	 * Updates the Candidate List according to the current solution
	 * {@link #sol}. In other words, this method is responsible for
	 * updating which elements are still viable to take part into the solution.
	 */
	public abstract void updateCL();

	/**
	 * Creates a new solution which is empty, i.e., does not contain any
	 * element.
	 * 
	 * @return An empty solution.
	 */
	public abstract Solution<E> createEmptySol();

	/**
	 * The GRASP local search phase is responsible for repeatedly applying a
	 * neighborhood operation while the solution is getting improved, i.e.,
	 * until a local optimum is attained.
	 * 
	 * @return An local optimum solution.
	 */
	public abstract Solution<E> localSearch();

	/**
	 * Constructor for the AbstractGRASP class.
	 * 
	 * @param objFunction
	 *            The objective function being minimized.
	 * @param alpha
	 *            The GRASP greediness-randomness parameter (within the range
	 *            [0,1])
	 * @param iterations
	 *            The number of iterations which the GRASP will be executed.
	 */
	public AbstractGRASP(Evaluator<E> objFunction, Double alpha, Integer iterations) {
		this.ObjFunction = objFunction;
		this.alpha = alpha;
		this.iterations = iterations;
	}
	
	/**
	 * The GRASP constructive heuristic, which is responsible for building a
	 * feasible solution by selecting in a greedy-random fashion, candidate
	 * elements to enter the solution.
	 * 
	 * @return A feasible solution to the problem being minimized.
	 */
	public Solution<E> constructiveHeuristic(ConstructiveMethod method, String... args) {
		switch (method) {
			case STANDARD:
				return standardConstructiveHeuristic();
			case RANDOM_PLUS_GREEDY:
				return randomPlusGreedyConstructiveHeuristic(args);
			case REACTIVE_GRASP:
			default:
				System.out.println("Method not implemented");
				return createEmptySol();
		}
	}

	private Solution<E> randomPlusGreedyConstructiveHeuristic(String... args) {
		CL = makeCL();
		RCL = makeRCL();
		sol = createEmptySol();
		cost = Double.POSITIVE_INFINITY;
		Integer p = Integer.parseInt(args[0]);
		Integer iteration = Integer.parseInt(args[1]);

		if (iteration > p) {
			if (iteration == p + 1) {
				System.out.println("(Iter. " + iteration + ") Started to use greedy");
			}
			/* Main loop, which repeats until the stopping criteria is reached. */
			while (!constructiveStopCriteria()) {

				E minCandidate = null;
				Double minCost = Double.POSITIVE_INFINITY;
				cost = ObjFunction.evaluate(sol);
				updateCL();

				/*
				 * Explore all candidate elements to enter the solution, saving the
				 * lowest cost variation achieved by the candidates.
				 */
				for (E c : CL) {
					Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
					if (deltaCost < minCost) {
						minCost = deltaCost;
						minCandidate = c;
					}
				}

				if (minCandidate != null) {
					CL.remove(minCandidate);
					sol.add(minCandidate);
					ObjFunction.evaluate(sol);
					RCL.clear();
				}
			}
			return sol;
		} else {
			return standardConstructiveHeuristic();
		}
	}

	private Solution<E> standardConstructiveHeuristic() {
		CL = makeCL();
		RCL = makeRCL();
		sol = createEmptySol();
		cost = Double.POSITIVE_INFINITY;

		/* Main loop, which repeats until the stopping criteria is reached. */
		while (!constructiveStopCriteria()) {

			double maxCost = Double.NEGATIVE_INFINITY, minCost = Double.POSITIVE_INFINITY;
			cost = ObjFunction.evaluate(sol);
			updateCL();

			/*
			 * Explore all candidate elements to enter the solution, saving the
			 * highest and lowest cost variation achieved by the candidates.
			 */
			for (E c : CL) {
				Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
				if (deltaCost < minCost)
					minCost = deltaCost;
				if (deltaCost > maxCost)
					maxCost = deltaCost;
			}

			/*
			 * Among all candidates, insert into the RCL those with the highest
			 * performance using parameter alpha as threshold.
			 */
			for (E c : CL) {
				Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
				if (deltaCost <= minCost + alpha * (maxCost - minCost)) {
					RCL.add(c);
				}
			}

			/* Choose a candidate randomly from the RCL */
			if (!RCL.isEmpty()) {
				int rndIndex = rng.nextInt(RCL.size());
				E inCand = RCL.get(rndIndex);
				CL.remove(inCand);
				sol.add(inCand);
				ObjFunction.evaluate(sol);
				RCL.clear();
			}
		}

		return sol;
	}

	/**
	 * The GRASP mainframe. It consists of a loop, in which each iteration goes
	 * through the constructive heuristic and local search. The best solution is
	 * returned as result.
	 * 
	 * @return The best feasible solution obtained throughout all iterations.
	 */
	public Solution<E> solve(ConstructiveMethod constructiveMethod, String... args) {
		int MAX_TIME_IN_SECONDS = 30 * 60; // 30 minutes

		Instant started = Instant.now();
		bestSol = createEmptySol();
		for (int i = 0; i < iterations; i++) {
			if (ConstructiveMethod.RANDOM_PLUS_GREEDY.equals(constructiveMethod)) {
				args[1] = String.valueOf(i);
			}
			constructiveHeuristic(constructiveMethod, args);
			localSearch();
			if (bestSol.cost > sol.cost) {
				bestSol = new Solution<E>(sol);
				if (verbose)
					System.out.println("(Iter. " + i + ") BestSol = " + bestSol);
			}
			if (Instant.now().getEpochSecond() > started.plusSeconds(MAX_TIME_IN_SECONDS).getEpochSecond()) {
				System.out.println("Interrupting");
				break;
			}
		}

		return bestSol;
	}

	/**
	 * A standard stopping criteria for the constructive heuristic is to repeat
	 * until the current solution improves by inserting a new candidate
	 * element.
	 * 
	 * @return true if the criteria is met.
	 */
	public Boolean constructiveStopCriteria() {
		return (cost > sol.cost) ? false : true;
	}

	public enum ConstructiveMethod {
		STANDARD, RANDOM_PLUS_GREEDY, REACTIVE_GRASP
	}

}
