<?php

namespace App\Http\Controllers;

use Symfony\Component\Process\Process;
use Illuminate\Http\Request;

class MainController extends Controller
{
    public function num(Request $request) {
        $dat = $request->input('dat');
        $dat = implode(' ', $dat);
        $p = new Process([
            config('backend.binPython'),
            'theai.py',
        ]);
        $p->setWorkingDirectory(base_path());
        $p->setInput($dat);

        $p->run();

        $out = $p->getOutput() ?: $p->getErrorOutput();
        $out = trim($out);

        return [
            'num' =>  $out,
        ];
    }
}
