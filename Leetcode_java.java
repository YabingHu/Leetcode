//Dynamic Programming:
//70. Climbing Stairs
// Time and Space O(n) for all methods
//Recursive
class Solution {
    public int climbStairs(int n) {
        int[] res=new int[n+1];
        return helper(n,res);
    }
    public int helper(int n , int[] res){
        if(n<=1) return 1;
        if(res[n]>0) return res[n];
        res[n]=helper(n-1,res) + helper(n-2,res);
        return res[n];
    }
}
//iterative    
class Solution {
    public int climbStairs(int n) {
        int[] dp=new int[n+1];
        dp[0]=1;
        dp[1]=1;
        for (int i = 2;i<=n;i++){
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n];
    }
}

//746. Min Cost Climbing Stairs
//Time =O(n), space=O(1)
class Solution {
    public int minCostClimbingStairs(int[] cost) {
        int dp1=0;
        int dp2=0;
        int res=0;
        for(int i=2;i<=cost.length;i++){
            res=Math.min(dp1+cost[i-1],dp2+cost[i-2]);
            dp2=dp1;
            dp1=res;
        }
        return res;  
    }
}

//space=O(n)
class Solution {
    public int minCostClimbingStairs(int[] cost) {
        int[] dp=new int[cost.length+1];
        for(int i=2;i<=cost.length;i++){
            dp[i]=Math.min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2]);
        }
        return dp[cost.length];
    }
}

//303. Range Sum Query
//Time O(n), space O(1)
class NumArray {
    int[] rangeSum;

    public NumArray(int[] nums) {
        rangeSum = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            rangeSum[i] = (i == 0) ? nums[i] : rangeSum[i - 1] + nums[i];
        }
    }
    
    public int sumRange(int i, int j) {
        return rangeSum[j] - ((i == 0) ? 0 : rangeSum[i - 1]);
    }
}

//53. Maximum Subarray
//Time=O(n), space=O(1)
class Solution {
    public int maxSubArray(int[] nums) {
        int res=Integer.MIN_VALUE;
        int temp=0;
        for(int i=0;i<nums.length;i++){
            temp=Math.max(nums[i],temp+nums[i]);
            res=Math.max(res,temp) ;
        }
        return res;
    }
}

//Dvivde and Conquer
//169. Majority Element
//Both time and space are O(n)
class Solution {
    public int majorityElement(int[] nums) {
        Map<Integer,Integer> dict=new HashMap<Integer,Integer>();
        for (int i=0;i<nums.length;i++){
            if(dict.containsKey(nums[i])){
                dict.put(nums[i], dict.get(nums[i])+1);
                if(dict.get(nums[i])>(nums.length)/2) return nums[i];
            }
            else dict.put(nums[i],1);
            
        }
        return nums[0];//if only contains one element
    }
}

//Linked List


//Graph
//133. Clone Graph
/*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> neighbors;

    public Node() {}

    public Node(int _val,List<Node> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/
//DFS solution, time and space O(n)
class Solution {
    public Node cloneGraph(Node node) {
        HashMap<Node,Node> map=new HashMap<>();
        return helper(node,map);
    }
    public Node helper(Node node,HashMap<Node,Node> map){
        if(node==null) return null;
        if(map.containsKey(node)) return map.get(node);
        Node dup=new Node(node.val,new ArrayList<Node>());
        map.put(node,dup);
        for (Node neighbor:node.neighbors){
            Node clone=helper(neighbor,map);
            dup.neighbors.add(clone);
        }
        return dup;
    }
}

//BFS solution, time and space O(n)
class Solution {
    public Node cloneGraph(Node node) {
        if(node==null) return null;
        HashMap<Node,Node> map=new HashMap<>();
        Queue<Node> queue = new LinkedList<>();
        queue.add(node);
        Node dup=new Node(node.val,new ArrayList<Node>());
        map.put(node,dup);
        while(!queue.isEmpty()){
            Node t=queue.poll();
            for(Node neighbor : t.neighbors){
                 if(!map.containsKey(neighbor)){
                     map.put(neighbor,new Node(neighbor.val,new ArrayList<Node>()));
                     queue.add(neighbor);
                 }
                map.get(t).neighbors.add(map.get(neighbor));
            }
        }
        return dup;
    }
}

//138. Copy List with Random Pointer
//Time:O(n), space:O(n)
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Map<Node, Node> map = new HashMap<Node, Node>();
        Node node = head;
        // loop 1. copy all the nodes
        while (node != null) {
            map.put(node, new Node(node.val));
            node = node.next;
        }
         // loop 2. assign next and random pointers
         node = head;
         while (node != null) {
             map.get(node).next = map.get(node.next);
             map.get(node).random = map.get(node.random);
             node = node.next;
         }
         return map.get(head);
    }
}




//200. Number of Islands
//Time and space O(m*n)
class Solution {
    public int numIslands(char[][] grid) {
        if(grid.length==0) return 0;
        int m=grid.length, n=grid[0].length,res=0;
        int[][] visited=new int[m][n];
        for (int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]=='1' && visited[i][j]==0){
                    helper(grid,visited,i,j);
                    res++;
                }
            }
        } 
    return res;
    }
    
    public void helper(char[][] grid,int[][] visited,int x,int y){
        if(x<0 ||x>= grid.length||y<0||y>=grid[0].length||grid[x][y]=='0'||visited[x][y]==1) return;
        visited[x][y]=1;
        helper(grid,visited,x-1,y);
        helper(grid,visited,x+1,y);
        helper(grid,visited,x,y-1);
        helper(grid,visited,x,y+1);
    }
}

//547. Friend Circles
//Time=O(n^2), space=O(n)
class Solution {
    public int findCircleNum(int[][] M) {
        if(M.length==0) return 0;
        int m=M.length;
        int[] visited=new int[m];
        int res=0;
        for (int i=0;i<m;i++){
            if (visited[i]==1) continue;
            helper(M,i,m,visited);
            res++;
        }
        return res;
    }
    public void helper(int[][] M,int i ,int m,int[] visited){
            if(visited[i]==1) return;
            visited[i]=1;
            for(int j=0;j<m;j++){
                if(M[i][j]==1 && visited[j]==0)
                helper(M,j,m,visited);
            }
    }
}

//733. Flood Fill
//Time=O(m*n),space=O(1)
class Solution {
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        if(image[sr][sc]==newColor) return image;
        int m=image.length;
        int n=image[0].length;
        helper(image,sr,sc,image[sr][sc],newColor);
        return image;
    }
    public void helper(int[][] image,int x, int y, int preColor, int newColor){
        int m=image.length;
        int n=image[0].length;
        if(x<0||x>=m||y<0||y>=n) return;
        if(image[x][y]!=preColor) return;
        image[x][y]=newColor;
        helper(image,x+1,y,preColor,newColor);
        helper(image,x-1,y,preColor,newColor);
        helper(image,x,y+1,preColor,newColor);
        helper(image,x,y-1,preColor,newColor);
    }
}

//Binary Search
//35. Search Insert Position
//Time=O(logn),sapce=O(1)
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left=0;
        int right=nums.length-1;
        while (left<=right){
            int mid=left+(right-left)/2;
            if(target>nums[mid]) left=mid+1;
            else right=mid-1;
        }
        return right+1;
    }
}

//34. Find First and Last Position of Element in Sorted Array
//Time=O(logn),sapce=O(1)
//Two binary search solution
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] res=new int[2];
        res[0]=-1;
        res[1]=-1;
        if(nums.length==0) return res;
        int left=0;
        int right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]<target) left=mid+1;
            else right=mid-1;
        }
        if(right+1==nums.length) return res;
        if(nums[right+1]!=target) return res;
        res[0]=right+1;
        right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]<=target) left=mid+1;
            else right=mid-1;
        }
        res[1]=right;
        return res;
    }
}

//704. Binary Search
//Time=O(logn),sapce=O(1)
class Solution {
    public int search(int[] nums, int target) {
        int left=0;
        int right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if (nums[mid]==target) return mid;
            else if (nums[mid]>target) right=mid-1;
            else left=mid+1;
        }
        return -1;
    }
}

//33. Search in Rotated Sorted Array
//Time=O(logn),sapce=O(1)
class Solution {
    public int search(int[] nums, int target) {
        int left=0;
        int right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]==target) return mid;
            else if (nums[mid]<nums[right]){
                if(nums[mid]<target && nums[right]>=target) left=mid+1;
                else right=mid-1;
            }
            else{
                if(nums[mid]>target&& nums[left]<=target) right=mid-1;
                else left=mid+1;
            }
        }
        return -1;
    }
}

//Tree:
//94. Binary Tree Inorder Traversal
//Time=O(n), space=O(n) for all methods.
//Recursive solution
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res =new ArrayList<Integer>();
        if(root==null) return res;
        helper(root,res);
        return res;
    }
    public void helper(TreeNode root, List<Integer>res){
        if(root==null) return;
        helper(root.left,res);
        res.add(root.val);
        helper(root.right,res);
    }
}
//Using stack
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res=new ArrayList<Integer>();
        Stack<TreeNode> s = new Stack<TreeNode>();
        if (root ==null) return res;
        TreeNode p =root;
        while(p!=null || ! s.isEmpty()){
            while(p!=null){
                s.push(p);  
                p=p.left;
            }
            p=s.peek();
            s.pop();
            res.add(p.val);
            p=p.right;
        }
        return res; 
    }
}

//144. Binary Tree Preorder Traversal
//Time O(n), space=O(n) for both methods
//Iterative:    
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        Stack<TreeNode> s = new Stack<TreeNode>();
        if (root==null) return res;
        s.push(root);
        while(!s.isEmpty()){
            TreeNode cur=s.pop();
            res.add(cur.val);
            if(cur.right!=null) s.push(cur.right);
            if(cur.left!=null) s.push(cur.left);
        }
        return res;
    }
}

//Recursive:
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        if (root==null) return res;
        helper(root,res);
        return res;
    }
    public void helper(TreeNode root, List<Integer> res){
        if(root==null) return;
        res.add(root.val);
        helper(root.left,res);
        helper(root.right,res);
    }
}

//589. N-ary Tree Preorder Traversal
//Time O(n), space=O(n) for both methods
//Iterative
class Solution {
    public List<Integer> preorder(Node root) {
        List<Integer> res = new ArrayList<Integer>();
        if (root==null) return res;
        Stack<Node> s=new Stack<Node>();
        s.push(root);
        while (! s.isEmpty()){
            Node cur=s.pop();
            res.add(cur.val);
            for(int i=cur.children.size()-1;i>=0;i--){
                s.push(cur.children.get(i));
            }
        }
        return res;
    }
}

//Recursive
class Solution {
    public List<Integer> preorder(Node root) {
        List<Integer> res = new ArrayList<Integer>();
        if (root==null) return res;
        helper(root,res);
        return res;
    }
    public void helper(Node root, List<Integer>res){
        if(root==null) return;
        res.add(root.val);
        for(Node child : root.children){
            helper(child,res);
        }
    }
}

//145. Binary Tree Postorder Traversal
//Time=O(n), space=O(n)
//Iterative
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res=new ArrayList<Integer>();
        if(root==null) return res;
        Stack<TreeNode> s = new Stack<TreeNode>();
        s.push(root);
        TreeNode head=root;
        while(!s.isEmpty()){
            TreeNode t=s.pop();
            res.add(0,t.val);
            if(t.left!=null) s.push(t.left);
            if(t.right!=null) s.push(t.right);
        }
            
        return res;
    }
}
//Recursive
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root==null) return res;
        helper(res,root);
        return res;
    }
    public static void helper(List<Integer>res, TreeNode root){
        if(root==null) return;
        helper(res,root.left);
        helper(res,root.right);
        res.add(root.val);
    }
}
//590. N-ary Tree Postorder Traversal
//Time=O(n), space=O(n)
class Solution {
    public List<Integer> postorder(Node root) {
        List<Integer> res = new ArrayList<Integer>();
        if (root==null) return res;
        Stack<Node> s=new Stack<Node>();
        s.push(root);
        while (! s.isEmpty()){
            Node cur=s.pop();
            res.add(cur.val);
            for(Node child:cur.children){
                s.push(child);
            }
        }
        Collections.reverse(res);
        return res;
    }
}

//46. Permutations
//Time=O(n!), space=O(n!)
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        helper(nums,0,res);
        return res;
    }
     public static void helper(int[]nums,int start, List<List<Integer>> res) {
         if (start == nums.length) {
            List<Integer> temp = new ArrayList<>();
            for(int k = 0; k < nums.length; k++){
            temp.add(nums[k]);}
            res.add(temp);
            return;
        }
        for (int i = start; i < nums.length; ++i) {
            int tmp = nums[start];
            nums[start] = nums[i];
            nums[i] = tmp;
            helper(nums, start + 1, res);
            tmp = nums[start];
            nums[start] = nums[i];
            nums[i] = tmp;
        }
         
    }
};

//Another mothod
//Time=O(n!), space=O(n!)
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res =new ArrayList<List<Integer>>();
        List<Integer> out = new ArrayList<Integer>();
        List<Integer> visited = new ArrayList<Integer>();
        for (int i = 0; i < nums.length; i++) {
          visited.add(0);
        }
        helper(nums,0,visited,out,res);
        return res;
    }
    public void helper(int[] nums,int start,List<Integer> visited,List<Integer> out,List<List<Integer>> res){
        if(nums.length==start){
            res.add(new ArrayList<Integer>(out));
            return;
        }
        for (int i=0;i<nums.length;i++){
            if(visited.get(i)==1) continue;
            visited.set(i,1);
            out.add(nums[i]);
            helper(nums,start+1,visited,out,res);
            out.remove(out.size()-1);
            visited.set(i,0);
        }
        
    }
}

//47. Permutations II
//Time=O(n!), space=O(n!)
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res =new ArrayList<List<Integer>>();
        List<Integer> out = new ArrayList<Integer>();
        List<Integer> visited = new ArrayList<Integer>();
        for (int i = 0; i < nums.length; i++) {
          visited.add(0);
        }
        Arrays.sort(nums); 
        helper(nums,0,visited,out,res);
        return res;
    }
    public void helper(int[] nums,int start,List<Integer> visited,List<Integer> out,List<List<Integer>> res){
        if(nums.length==start){
            res.add(new ArrayList<Integer>(out));
            return;
        }
        for (int i=0;i<nums.length;i++){
            if(visited.get(i)==1) continue;
            if(i>0 && nums[i]==nums[i-1]&& visited.get(i-1)==0) continue;
            visited.set(i,1);
            out.add(nums[i]);
            helper(nums,start+1,visited,out,res);
            out.remove(out.size()-1);
            visited.set(i,0);
        }
        
    }
}

//784. Letter Case Permutation
//Time=O(n* 2^l, l the number of letters in a string), space=O(n)+O(n*2^l)
class Solution {
    public List<String> letterCasePermutation(String S) {
    List<String> ans = new ArrayList<>();
    dfs(S.toCharArray(), 0, ans);
    return ans;
  }
  
  public void dfs(char[] S, int i, List<String> ans) {
    if (i == S.length) {
      ans.add(new String(S));
      return;
    }    
    dfs(S, i + 1, ans);    
    if (!Character.isLetter(S[i])) return;
    S[i] ^= 1 << 5;
    dfs(S, i + 1, ans);
    S[i] ^= 1 << 5;
  }
}
