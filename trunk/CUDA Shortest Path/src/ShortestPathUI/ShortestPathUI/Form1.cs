using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Diagnostics;
using System.Threading;
using System.IO;

namespace ShortestPathUI
{
    public partial class Form1 : Form
    {
        string filename = "";
        string CPUProg = "";
        string GPUProg = "";
        Process p;
        StreamReader OutReader;
        bool finishRead;
        string progOutput;

        public Form1()
        {
            InitializeComponent();
        }

        private void DisplayOutput()
        {
            finishRead = false;
            progOutput = OutReader.ReadLine();
            finishRead = true;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog fileDialog = new OpenFileDialog();
            string defaultGraphPath = "c:\\graph";
            if (Directory.Exists(defaultGraphPath))
                fileDialog.InitialDirectory = defaultGraphPath;
            if (fileDialog.ShowDialog() == DialogResult.OK)
                filename = fileDialog.FileName;
            label4.Text = System.IO.Path.GetFileName(filename);

            // 处理节点范围
            System.IO.StreamReader fin = System.IO.File.OpenText(filename);
            string line;
            string n = "";
            while ((line = fin.ReadLine()).Length != 0)
            {
                string[] wordline = line.Split();

                if (wordline[0] == "p")
                {
                    n = wordline[2];
                    break;
                }
            }
            fin.Close();

            textBox1.Text = "1";
            textBox2.Text = n;

            label11.Text = "1 到 " + n;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            OpenFileDialog fileDialog = new OpenFileDialog();
            string CPUPath = @"C:\Users\ChenKai\Documents\Visual Studio 2010\Projects\CPU\ShortestPath\Release";
            if (Directory.Exists(CPUPath))
                fileDialog.InitialDirectory = CPUPath;
            if (fileDialog.ShowDialog() == DialogResult.OK)
                CPUProg = fileDialog.FileName;
            label6.Text = System.IO.Path.GetFileName(CPUProg);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            OpenFileDialog fileDialog = new OpenFileDialog();
            string GPUPath = @"C:\Users\ChenKai\Documents\Visual Studio 2010\Projects\GPU\ShortestPath\x64\Release";
            if (Directory.Exists(GPUPath))
                fileDialog.InitialDirectory = GPUPath;
            if (fileDialog.ShowDialog() == DialogResult.OK)
                GPUProg = fileDialog.FileName;
            label9.Text = System.IO.Path.GetFileName(GPUProg);
        }

        private string GetResult()
        {
            progOutput = "";
            Thread threadOutput = new Thread(new ThreadStart(DisplayOutput));

            threadOutput.Start();

            while ((progOutput == "") || (!p.HasExited && !finishRead))
                Thread.Sleep(10);

            if (!finishRead)
                p.Kill();

            return progOutput;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string prog = "";
            string type = "";
            if (radioButton1.Checked)
            {
                if (comboBox1.SelectedIndex == -1)
                {
                    MessageBox.Show("未选择CPU的算法!");
                    return;
                }

                if (label6.Text == "未选")
                {
                    MessageBox.Show("未选择CPU程序！");
                    return;
                }

                prog = CPUProg;
                type = comboBox1.SelectedIndex.ToString();
            }
            else if (radioButton2.Checked)
            {
                if (comboBox2.SelectedIndex == -1)
                {
                    MessageBox.Show("未选择GPU的算法!");
                    return;
                }

                if (label9.Text == "未选")
                {
                    MessageBox.Show("未选择GPU程序！");
                    return;
                }
                prog = GPUProg;
                type = comboBox2.SelectedIndex.ToString();
            }

            string source = textBox1.Text;
            string target = textBox2.Text;

            p = new Process();
            p.StartInfo.FileName = prog;
            p.StartInfo.Arguments = type + " " + filename + " " + source + " " + target;
            if (radioButton2.Checked && checkBox1.Checked)
                p.StartInfo.Arguments += " 1 ";
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.RedirectStandardInput = true;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.RedirectStandardError = true;
            p.StartInfo.CreateNoWindow = true;

            try
            {
                p.Start();
            }
            catch
            {
                MessageBox.Show("程序启动失败！");
                return;
            }

            textBox5.Text = p.StandardOutput.ReadToEnd();

            p.WaitForExit();

            p.Close();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }
    }
}
