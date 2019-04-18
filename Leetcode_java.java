//Dynamic Programming:
//70. Climbing Stairs
// Time and Space O(n) for all methods


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

