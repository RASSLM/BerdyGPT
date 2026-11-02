using System;

namespace BerdyGPT
{
    public class Separator
    {
        public string[] Separate(string line)
        {
            string[] parts = line.Split("<|SEP|>");

            parts[0] = parts[0].Trim();
            parts[1] = parts[1].Trim();

            return parts;
        }
    }
}