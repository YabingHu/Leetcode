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
  
#746. Min Cost Climbing Stairs
# Time and Space O(n) for all methods
# Or space O(1)

#Itrative,space O(1)    
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        res=0
        dp1,dp2=0,0
        for i in range(2,len(cost)+1):
            dp=min(dp1+cost[i-1],dp2+cost[i-2])
            dp2=dp1
            dp1=dp
        return dp
    
#Itrative,space O(n)    
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        res=0
        dp=[0]*(len(cost)+1)
        dp[0],dp[1]=0,0
        for i in range(2,len(cost)+1):
            dp[i]=min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])
        return dp[len(cost)]        
       
#303. Range Sum Query - Immutable   
class NumArray:
    def __init__(self, nums: List[int]):
        self.dp = nums
        for i in range(1,len(nums)):
            self.dp[i]+=self.dp[i-1]
    def sumRange(self, i: int, j: int) -> int:
        if i==0:
            return self.dp[j]
        else:
            return self.dp[j]-self.dp[i-1]
        
#53. Maximum Subarray    


#121. Best Time to Buy and Sell Stock
#Time O(n),space O(1)
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy=float('Inf')
        res=0
        for i in range(len(prices)):
            buy=min(prices[i],buy)
            res=max(res,prices[i]-buy)
        return res
    
#Divide and Conquer
#169. Majority Element

#Moore Voting time:O(n) space O(1)
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        res=0
        cnt=0
        for num in nums:
            if cnt==0:
                res=num
                cnt+=1
            else:
                if num==res:
                    cnt+=1
                else:
                    cnt-=1
        return res
        
        
