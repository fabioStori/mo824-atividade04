package problems.qbf.solvers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import metaheuristics.grasp.AbstractGRASP;
import problems.qbf.KQBF_Inverse;
import solutions.Solution;



/**
 * Metaheuristic GRASP (Greedy Randomized Adaptive Search Procedure) for
 * obtaining an optimal solution to a QBF (Quadractive Binary Function --
 * {@link #QuadracticBinaryFunction}). Since by default this GRASP considers
 * minimization problems, an inverse QBF function is adopted.
 * 
 * @author ccavellucci, fusberti
 */
public class GRASP_KQBF extends AbstractGRASP<Integer> {

	/**
	 * KQBFInverse obj function
	 */
	public KQBF_Inverse KQBFInverse;

	public List<Integer> allCandidateList;

	public boolean useFirstImprove;

	/**
	 * Constructor for the GRASP_QBF class. An inverse QBF objective function is
	 * passed as argument for the superclass constructor.
	 * 
	 * @param alpha
	 *            The GRASP greediness-randomness parameter (within the range
	 *            [0,1])
	 * @param iterations
	 *            The number of iterations which the GRASP will be executed.
	 * @param KQBFInverse
	 *            KQBF_ Inverse Evaluator.
	 * @throws IOException
	 *             necessary for I/O operations.
	 */
	public GRASP_KQBF(Double alpha, Integer iterations, Integer maxTimeInSeconds, boolean useFirstImprove, KQBF_Inverse KQBFInverse) throws IOException {
		super(KQBFInverse, alpha, iterations, maxTimeInSeconds);
		this.KQBFInverse = KQBFInverse;
		this.allCandidateList = makeCL();
		this.useFirstImprove = useFirstImprove;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see grasp.abstracts.AbstractGRASP#makeCL()
	 */
	@Override
	public ArrayList<Integer> makeCL() {

		ArrayList<Integer> _CL = new ArrayList<Integer>();

		for (int i = 0; i < ObjFunction.getDomainSize(); i++) {
			Integer cand = i;
			_CL.add(cand);
		}

		return _CL;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see grasp.abstracts.AbstractGRASP#makeRCL()
	 */
	@Override
	public ArrayList<Integer> makeRCL() {

		ArrayList<Integer> _RCL = new ArrayList<Integer>();

		return _RCL;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see grasp.abstracts.AbstractGRASP#updateCL()
	 */
	@Override
	public void updateCL() {
		Double[] weights = KQBFInverse.getWeights();
		Double freeCapacity = KQBFInverse.getCapacity() - sol.usedCapacity;

		/*
		* Select only viable candidates given the free capacity and update the CL.
		* */
		ArrayList<Integer> newCL = new ArrayList<>();
		for (Integer candidate : allCandidateList) {
			if (!sol.contains(candidate) && weights[candidate] <= freeCapacity) {
				newCL.add(candidate);
			}
		}
		CL = newCL;
	}

	/**
	 * {@inheritDoc}
	 * 
	 * This createEmptySol instantiates an empty solution and it attributes a
	 * zero cost, since it is known that a QBF solution with all variables set
	 * to zero has also zero cost.
	 */
	@Override
	public Solution<Integer> createEmptySol() {
		Solution<Integer> sol = new Solution<Integer>();
		sol.cost = 0.0;
		sol.usedCapacity = 0.0;
		return sol;
	}

	/**
	 * {@inheritDoc}
	 * 
	 * The local search operator developed for the QBF objective function is
	 * composed by the neighborhood moves Insertion, Removal and 2-Exchange.
	 */
	@Override
	public Solution<Integer> localSearch() {
		return useFirstImprove ? firstImproving() : bestImproving();
	}

	private Solution<Integer> firstImproving() {
		Double minDeltaCost;
		Integer bestCandIn = null, bestCandOut = null;

		do {
			minDeltaCost = Double.POSITIVE_INFINITY;
			updateCL();

			// Evaluate insertions
			for (Integer candIn : CL) {
				double deltaCost = ObjFunction.evaluateInsertionCost(candIn, sol);
				if (deltaCost < -Double.MIN_VALUE) {
					minDeltaCost = deltaCost;
					bestCandIn = candIn;
					bestCandOut = null;
					break;
				}
			}

			if (bestCandIn == null) {
				// Evaluate removals
				for (Integer candOut : sol) {
					double deltaCost = ObjFunction.evaluateRemovalCost(candOut, sol);
					if (deltaCost < -Double.MIN_VALUE) {
						minDeltaCost = deltaCost;
						bestCandIn = null;
						bestCandOut = candOut;
						break;
					}
				}

				if (bestCandOut == null) {
					// Evaluate exchanges
					for (Integer candIn : CL) {
						for (Integer candOut : sol) {
							double deltaCost = ObjFunction.evaluateExchangeCost(candIn, candOut, sol);
							if (deltaCost < -Double.MIN_VALUE) {
								minDeltaCost = deltaCost;
								bestCandIn = candIn;
								bestCandOut = candOut;
								break;
							}
						}
					}
				}
			}
			// Implement the best move, if it reduces the solution cost.
			if (minDeltaCost < -Double.MIN_VALUE) {
				if (bestCandOut != null) {
					sol.remove(bestCandOut);
					CL.add(bestCandOut);
				}
				if (bestCandIn != null) {
					sol.add(bestCandIn);
					CL.remove(bestCandIn);
				}
				ObjFunction.evaluate(sol);
			}
		} while (minDeltaCost < -Double.MIN_VALUE);

		return null;
	}

	private Solution<Integer> bestImproving() {
		Double minDeltaCost;
		Integer bestCandIn = null, bestCandOut = null;

		do {
			minDeltaCost = Double.POSITIVE_INFINITY;
			updateCL();

			// Evaluate insertions
			for (Integer candIn : CL) {
				double deltaCost = ObjFunction.evaluateInsertionCost(candIn, sol);
				if (deltaCost < minDeltaCost) {
					minDeltaCost = deltaCost;
					bestCandIn = candIn;
					bestCandOut = null;
				}
			}
			// Evaluate removals
			for (Integer candOut : sol) {
				double deltaCost = ObjFunction.evaluateRemovalCost(candOut, sol);
				if (deltaCost < minDeltaCost) {
					minDeltaCost = deltaCost;
					bestCandIn = null;
					bestCandOut = candOut;
				}
			}
			// Evaluate exchanges
			for (Integer candIn : CL) {
				for (Integer candOut : sol) {
					double deltaCost = ObjFunction.evaluateExchangeCost(candIn, candOut, sol);
					if (deltaCost < minDeltaCost) {
						minDeltaCost = deltaCost;
						bestCandIn = candIn;
						bestCandOut = candOut;
					}
				}
			}
			// Implement the best move, if it reduces the solution cost.
			if (minDeltaCost < -Double.MIN_VALUE) {
				if (bestCandOut != null) {
					sol.remove(bestCandOut);
					CL.add(bestCandOut);
				}
				if (bestCandIn != null) {
					sol.add(bestCandIn);
					CL.remove(bestCandIn);
				}
				ObjFunction.evaluate(sol);
			}
		} while (minDeltaCost < -Double.MIN_VALUE);

		return null;
	}

	/**
	 * A main method used for testing the GRASP metaheuristic.
	 * 
	 */
	public static void main(String[] args) throws IOException {

		long startTime = System.currentTimeMillis();
		KQBF_Inverse QBF_Inverse = new KQBF_Inverse("instances/kqbf/kqbf020");
		double alpha = 0.05;
		int iterations = 1000;
		int maxTimeInSeconds = 30 * 60; // 30 minutes
		boolean useFirstImprove = false;
		ConstructiveMethod method = ConstructiveMethod.RANDOM_PLUS_GREEDY;

		GRASP_KQBF grasp = new GRASP_KQBF(alpha, iterations, maxTimeInSeconds, useFirstImprove, QBF_Inverse);
		Solution<Integer> bestSol;
		System.out.println(
				"Running method: " + method +
				". Alpha: " + (method.name().contains("REACTIVE_GRASP") ? "N/A" : alpha) +
				". Max iterations: " + iterations +
				". Max time in seconds: " + maxTimeInSeconds +
				". Use first-improving: " + useFirstImprove
		);
		switch (method) {
			case RANDOM_PLUS_GREEDY:
				String randomizedThreshold = "3";
				String currentIteration = "0";
				bestSol = grasp.solve(ConstructiveMethod.RANDOM_PLUS_GREEDY, randomizedThreshold, currentIteration);
				break;
			case RANDOM_REACTIVE_GRASP:
			case BEST_ALPHA_REACTIVE_GRASP:
			case STANDARD:
				bestSol = grasp.solve(method);
				break;
			default:
				System.out.println("Method " + method + " not implemented.");
				bestSol = null;
		}

		System.out.println("alpha " + alpha + ", iterations " + iterations + ", maxVal = " + bestSol);
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Time = "+(double)totalTime/(double)1000+" seg");

	}

}
