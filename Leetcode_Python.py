#Dynamic Programming:

#70. Climbing Stairs
# Time and Space O(n) for all methods

#recursive
class Solution:
    def climbStairs(self, n: int) -> int:
        res=[0]*(n+1)
        return self.helper(n,res)

    def helper(self,n,res):
        if(n<=1): return 1
        if res[n]>0: return res[n]
        res[n]=self.helper(n-1,res)+self.helper(n-2,res)
        return res[n]
#iterative       
class Solution:
    def climbStairs(self, n: int) -> int:
        dp=[0]*(n+1)
        dp[0],dp[1]=1,1
        for i in range(2,n+1):
            dp[i]=dp[i-1]+dp[i-2]
        return dp[n]
        
