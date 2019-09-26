#826. Most Profit Assigning Work
#Time = O(NlogN+QlogQ), where N is the number of jobs, and Q is the number of people，Space=O(N)

class Solution:
    def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:
        jobs = sorted(zip(difficulty, profit))
        res = 0
        i = 0
        best = 0
        for ability in sorted(worker):
            while i < len(jobs) and ability >= jobs[i][0]:
                best = max(jobs[i][1], best)
                i += 1
            res += best
        return res
