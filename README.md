**Title**
Network Friendly Recommendations Project: 1st assignment

---

**Description**
As mentioned in class, this assignment is basically "part 1" of your final project assignment. The idea is to implement a simple(r) version of the problem that can be solved with "exact" methods (i.e., no approximations based on neural networks or other methods) that we learned in the first lectures abour MDP and Q-learning methods. I provide below my suggestions for implementing this first version of your project:

---

**Environment (Content Catalogue):**

* **K content items.** I would pick K <= 100 initially, as the code might get pretty slow otherwise.
* **For every pair of items, i and j,** create a random value u\_ij in \[0,1] that indicates how related content j is to content i. This i basically a random array U of size K-by-K (you can choose to make it symmetric or not). Assume the diagonal of this array (i.e. all elements u\_ii) are equal to 0 (to avoid recommending the same item just watched).
* **Assume there is a threshold u\_min in \[0,1]** below which two contents are "irrelevant" (this is an input parameter to play around with and observe its impact).
* **Assume C out of these contents are cached** (use values of C <= 0.2 K). Cached items have cost 0, and non-cached have cost 1.

---

**Environment (User Model):**

* A user might watch multiple items, one after the other during a session.
* After a user watches a content, **N = 2** new items are recommended.
* With probability **q** (input parameter): she ends the viewing session (i.e. does not watch another video)
* With probability **1-q,** she proceeds with one more video as follows:

  * **If ALL N recommended are "relevant"** (i.e., have higher u\_ij than u\_min), then

    * with probability **α** (input parameter): the user picks one of the N recommended items (with equal probability)
    * With probability **1-α:** the users picks any item k from the entire catalogue of K items with probability p\_k (you can assume for simplicity that p\_k = 1/K....i.e. uniform).
  * **If at least one of the N recommendations is irrelevant,** then again the users picks any item k from the entire catalogue of K items with probability p\_k.

---

**Control variables and objective:**

* Your algorithm must learn/optimize: for every possible item i the user might watch, a tuple of N items to recommend.
* **Objective:** minimize the average total cost of items watched during a session.

---

**Algorithms to implement:**

* Policy Iteration to optimize recommendations, assuming all environment parameters are known.
* Q-learning, assuming parameters α and u\_min are not known (but the u\_ij relevance values, and q are still known).

---

**Experiments to try and report:**

1. Demonstrate that Policy Iteration indeed solves the problem optimally (you could try a hand-picked, toy scenario for this, not necessary randomly chosen).
2. Show the average cost achieved for different parameters (K, u\_min, a, q)
3. Prove that your Q-learning algorithm correctly learns the optimal policy as well (in the partially model-free environment above). Demonstrate the convergence speed to the optimal (for different K values).
4. Try to "break" Q-learning. At what values of K does your PC or colab code start going "too slow"?
